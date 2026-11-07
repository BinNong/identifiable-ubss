"""Create a Zenodo deposition for the reproduction package and backfill its DOI.

Workflow (publishing is deliberately a separate, explicit step):

    python experiments/publish_zenodo.py --check
    python experiments/publish_zenodo.py --dump-metadata
    ZENODO_TOKEN=... python experiments/publish_zenodo.py               # draft + upload
    ZENODO_TOKEN=... python experiments/publish_zenodo.py --publish     # irreversible
    python experiments/publish_zenodo.py --backfill 10.5281/zenodo.1234567

The draft mode creates an unpublished deposition with a prereserved DOI, uploads
the archive and prints the draft URL so the record can be reviewed in the browser
before publishing. Zenodo registers the DOI with DataCite only on publish, and a
published deposition can no longer be deleted.

Use --sandbox for a dry run on https://sandbox.zenodo.org (separate token,
10.5072 test DOI prefix, never backfilled into the manuscript).

Token scopes required: deposit:write and, for --publish, deposit:actions.
Create a token at https://zenodo.org/account/settings/applications/tokens/new/
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "signal-processing-submission.zip"
MANIFEST = PROJECT / "results" / "submission_manifest.json"
DECLARATIONS = PROJECT / "paper" / "declarations.tex"
CHECKLIST = PROJECT / "paper" / "submission_checklist.txt"
DOI_RECORD = PROJECT / "results" / "archive_doi.json"

PROD_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"
USER_AGENT = "identifiable-ubss/1.0 (+mailto:nongbin@tfswufe.edu.cn)"

TITLE = (
    "Reproduction package for: Joint identifiability of underdetermined "
    "post-nonlinear mixtures across regimes: Support coverage and intersection lattices"
)

DESCRIPTION = (
    "<p>Full reproducibility package accompanying the manuscript "
    "<em>Joint identifiability of underdetermined post-nonlinear mixtures across "
    "regimes: Support coverage and intersection lattices</em>.</p>"
    "<p>Contents: LaTeX sources and review PDFs of the 30-page main manuscript, "
    "15-page supplementary material, declarations, highlights and "
    "BibTeX references; source code under <code>src/</code> and "
    "<code>experiments/</code>; experiment configurations, raw and aggregated "
    "JSON results, exact binomial intervals, theory-verification and validation "
    "reports; generated vector figures and LaTeX result tables; and a SHA-256 "
    "manifest of every file in this archive.</p>"
    "<p>Reproduce with <code>python experiments/run_all.py --full --workers 6</code> "
    "(or <code>--smoke</code> for a reduced check) after installing "
    "<code>requirements.txt</code>. All numerical tables and figures are "
    "regenerated from the saved JSON results. The measured-speech experiment is "
    "semi-synthetic: activation blocks, mixing and post-nonlinear distortions "
    "are synthetic.</p>"
    "<p>Third-party material: three unmodified recordings "
    "(<code>0_jackson_0.wav</code>, <code>0_nicolas_0.wav</code>, "
    "<code>0_theo_0.wav</code>) from the "
    "<a href='https://github.com/Jakobovski/free-spoken-digit-dataset'>Free Spoken "
    "Digit Dataset</a> (Jakobovski and contributors), distributed under "
    "CC BY-SA 4.0, are included as measured source waveforms only. The archive "
    "is therefore released under CC BY-SA 4.0.</p>"
)

# Deposit metadata for https://zenodo.org/api/deposit/depositions
METADATA = {
    "title": TITLE,
    "upload_type": "software",
    "publication_date": "2026-11-07",
    "creators": [
        {
            "name": "Nong, Bin",
            "affiliation": "Tianfu College of Southwestern University of Finance and Economics, Mianyang, China",
        },
        {
            "name": "Fu, Weihong",
            "affiliation": "School of Telecommunications Engineering, Xidian University, Xi'an, China",
        },
    ],
    "description": DESCRIPTION,
    "access_right": "open",
    "license": "cc-by-sa-4.0",
    "version": "1.0",
    "language": "eng",
    "keywords": [
        "blind source separation",
        "post-nonlinear mixtures",
        "underdetermined mixtures",
        "identifiability",
        "sparse component analysis",
        "reproducibility",
    ],
    "prereserve_doi": True,
}

REQUIRED_FIELDS = ("title", "upload_type", "description", "creators", "access_right", "license")


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_local(metadata):
    problems = []
    if not ARCHIVE.is_file():
        problems.append(f"missing archive: {ARCHIVE}")
    if not MANIFEST.is_file():
        problems.append(f"missing manifest: {MANIFEST}")
    if ARCHIVE.is_file() and MANIFEST.is_file():
        recorded = json.loads(MANIFEST.read_text())
        actual = sha256(ARCHIVE)
        if recorded.get("sha256") != actual:
            problems.append(f"archive hash mismatch: manifest={recorded.get('sha256')} actual={actual}")
        else:
            print(f"archive ok      : {ARCHIVE.name} ({ARCHIVE.stat().st_size} bytes, {len(recorded.get('files', []))} files)")
            print(f"sha256          : {actual}")
    for field in REQUIRED_FIELDS:
        if not metadata.get(field):
            problems.append(f"missing metadata field: {field}")
    if metadata.get("access_right") in ("open", "embargoed") and not metadata.get("license"):
        problems.append("open access records require a license identifier")
    if metadata.get("license") not in ("cc-by-sa-4.0", "cc-by-4.0", "cc-zero"):
        problems.append(f"unexpected license id: {metadata.get('license')}")
    if ARCHIVE.is_file():
        import zipfile

        with zipfile.ZipFile(ARCHIVE) as bundle:
            wavs = [n for n in bundle.namelist() if n.endswith(".wav")]
            licenses = [n for n in bundle.namelist() if "license" in n.lower() or "copying" in n.lower()]
        if wavs and metadata.get("license") != "cc-by-sa-4.0":
            problems.append(
                "archive contains FSDD recordings (CC BY-SA 4.0) but license is "
                f"{metadata.get('license')}; use cc-by-sa-4.0 or repack without the WAV files"
            )
        if not licenses:
            print("note            : archive has no LICENSE file; the Zenodo record-level license applies")
        print(f"fsdd recordings : {len(wavs)}")
    print(f"metadata        : upload_type={metadata['upload_type']}, license={metadata['license']}, creators={len(metadata['creators'])}")
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        return False
    print("local checks    : PASS")
    return True


def api_request(method, url, token, payload=None, raw=None, content_type="application/json"):
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    if content_type:
        headers["Content-Type"] = content_type
    body = json.dumps(payload).encode("utf-8") if payload is not None else raw
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"{method} {url} -> HTTP {error.code}: {detail}") from error


def doi_sentence(doi):
    return (
        f"The full reproduction package is deposited at "
        f"\\href{{https://doi.org/{doi}}}{{doi:{doi}}} (Zenodo)."
    )


# Accepts both the current wording and the earlier "(Zenodo, version 1.0)" form,
# so backfilling an existing manuscript stays idempotent.
EXISTING_DOI_RE = re.compile(
    r"(?:A frozen, citable copy of the|The) full reproduction package is deposited at "
    r"\\href\{https://doi\.org/[^}]+\}\{doi:[^}]+\} \(Zenodo[^)]*\)\."
)


def backfill(doi):
    """Write the DOI into the Data availability statement and the checklist."""
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", doi):
        raise ValueError(
            f"refusing to backfill {doi!r}: only published production DOIs "
            "matching 10.5281/zenodo.<id> may go into the manuscript "
            "(sandbox 10.5072 DOIs are not registered with DataCite)"
        )
    text = DECLARATIONS.read_text()
    sentence = doi_sentence(doi)
    if EXISTING_DOI_RE.search(text):
        text = EXISTING_DOI_RE.sub(lambda _: sentence, text)
    else:
        anchor = "Source code is available from the \\href{https://github.com/BinNong/identifiable-ubss}{GitHub repository}."
        if anchor not in text:
            raise RuntimeError("could not locate the Data availability anchor sentence in declarations.tex")
        text = text.replace(anchor, anchor + " " + sentence, 1)
    DECLARATIONS.write_text(text)
    print(f"updated         : {DECLARATIONS.name}")

    lines = CHECKLIST.read_text().splitlines(keepends=True)
    old = "[ ] Ensure the GitHub repository is publicly accessible and deposit the full reproduction package in a permanent archive; add its DOI when issued."
    archive_line = f"[x] Deposit the full reproduction package in a permanent archive and cite its DOI (doi:{doi}) in the Data availability statement.\n"
    github_line = "[ ] Ensure the GitHub repository is publicly accessible (or replace the dead link in the Data availability statement).\n"
    replaced = False
    for index, line in enumerate(lines):
        if line.strip() == old:
            lines[index] = archive_line + github_line
            replaced = True
            break
        if line.startswith("[x] Deposit the full reproduction package") and "doi:10.5281/zenodo." in line:
            lines[index] = re.sub(r"doi:10\.5281/zenodo\.\d+", f"doi:{doi}", line)
            replaced = True
            break
    if not replaced:
        print("warning         : checklist archive line not found; DOI not written there", file=sys.stderr)
    CHECKLIST.write_text("".join(lines))
    print(f"updated         : {CHECKLIST.name}")

    DOI_RECORD.write_text(
        json.dumps(
            {
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}",
                "record_url": f"https://zenodo.org/records/{doi.rsplit('.', 1)[-1]}",
                "archive": ARCHIVE.name,
                "archive_sha256": sha256(ARCHIVE),
                "repository": "Zenodo",
                "version": "1.0",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote           : {DOI_RECORD.name}")
    print("reminder        : declarations.tex changed -> recompile main.pdf, then rebuild the zip and upload a new Zenodo version")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate local files and metadata only")
    parser.add_argument("--dump-metadata", action="store_true", help="print the deposit metadata JSON and exit")
    parser.add_argument("--publish", action="store_true", help="publish the deposition (IRREVERSIBLE, public)")
    parser.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org (test DOI prefix 10.5072)")
    parser.add_argument("--backfill", metavar="DOI", help="write an already-published DOI into the manuscript and exit")
    parser.add_argument("--license", dest="license_id", help="override the license id, e.g. cc-by-4.0")
    parser.add_argument("--token", help="Zenodo personal access token (default: env var)")
    args = parser.parse_args()

    metadata = dict(METADATA)
    if args.license_id:
        metadata["license"] = args.license_id

    if args.dump_metadata:
        print(json.dumps({"metadata": metadata}, indent=2, ensure_ascii=False))
        return 0
    if args.backfill:
        backfill(args.backfill.strip())
        return 0
    if not check_local(metadata):
        return 1
    if args.check:
        return 0

    base = SANDBOX_API if args.sandbox else PROD_API
    token_var = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
    token = args.token or os.environ.get(token_var)
    if not token:
        print(f"ERROR: no token. Set {token_var} or pass --token.", file=sys.stderr)
        print("       Create one at https://zenodo.org/account/settings/applications/tokens/new/", file=sys.stderr)
        print(f"       Scopes: deposit:write{', deposit:actions' if args.publish else ''}", file=sys.stderr)
        return 2

    print(f"creating draft  : {base}/deposit/depositions")
    deposition = api_request("POST", f"{base}/deposit/depositions", token, payload={"metadata": metadata})
    dep_id = deposition["id"]
    bucket = deposition["links"]["bucket"]
    reserved = deposition.get("metadata", {}).get("prereserved_doi", {}).get("doi")
    print(f"deposition id   : {dep_id}")
    if reserved:
        print(f"prereserved DOI : {reserved} (NOT active until published)")
    print(f"draft URL       : {base.replace('/api', '')}/deposit/{dep_id}")

    print(f"uploading       : {ARCHIVE.name} ({ARCHIVE.stat().st_size} bytes)")
    api_request("PUT", f"{bucket.rstrip('/')}/{ARCHIVE.name}", token, raw=ARCHIVE.read_bytes(), content_type="application/octet-stream")
    print("upload          : done")

    if not args.publish:
        print("\nDraft created but NOT published. Review it in the browser, then EITHER:")
        print("  1) click Publish on the Zenodo page, then run:")
        print("       python3 experiments/publish_zenodo.py --backfill <DOI>")
        print("  2) discard this draft and do everything in one command:")
        print("       ZENODO_TOKEN=... python3 experiments/publish_zenodo.py --publish")
        print("     (this creates a NEW deposition; it does not reuse the draft above)")
        print("Publishing registers the DOI with DataCite and cannot be undone.")
        return 0

    print("WARNING: publishing is irreversible; the record becomes public and cannot be deleted.")
    published = api_request("POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", token)
    doi = published.get("doi") or reserved
    if not doi:
        raise RuntimeError("publish succeeded but no DOI was returned; inspect the deposition in the browser")
    print(f"PUBLISHED DOI   : {doi}")
    print(f"record URL      : {published.get('record_url') or published.get('doi_url')}")
    if not args.sandbox:
        backfill(doi)
    else:
        print("sandbox DOI is a test identifier (10.5072) and was NOT written into the manuscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
