"""Build the grid's table body from the final_eval JSONs.

The rows were transcribed by hand up to now. That was survivable for one panel
and is not for three: 24 arms times four metrics is 96 numbers, each of which
has to carry the right seed pair, the right baseline and the right bold test.

Run with --check to reproduce the panels already in ddrl_grid.tex and diff
against it, which is what makes the generated FLUX rows trustworthy: the
conventions here (seed averaging, the standard error of a difference, the two
SE bold threshold) are inferred from those panels, so they have to reproduce
them exactly before the same code is used on new data.

Usage:
  python build_grid.py --check          compare generated rows to ddrl_grid.tex
  python build_grid.py --panel flux     print one panel's rows
"""
import argparse
import json
import os
import re

# Where the scored arms live. Defaults to evals/ beside this file; point it at
# a copy pulled off the cluster with
#   scp sc:/atlas/u/srijit/'final_eval*.json' report/grid/evals/
EVAL_DIR = os.environ.get(
    "EVAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
METRICS = ["clip_score", "pickscore", "imagereward"]

PANELS = {
    "sd35_ocr": {
        "header": r"\textbf{SD3.5-Medium, OCR}",
        "base": "base",
        "reward": "ocr",
        "rows": [
            (r"no anchor ($\beta=0$)   ", "swb042", "swb043"),
            (r"DDRL \; 0.0003          ", "swd000342", "swd000343"),
            (r"DDRL \; 0.001           ", "swd00142", "swd00143"),
            (r"DDRL \; 0.003           ", "swd00342", "swd00343"),
            (r"DDRL \; 0.01            ", "swd0142", "swd0143"),
            (r"FlowGRPO \; 0.01        ", "swf00142", "swf00143"),
            (r"FlowGRPO \; 0.04        ", "swf00442", "swf00443"),
            (r"FlowGRPO \; 0.1         ", "swf0142", "swf0143"),
        ],
    },
    "sd35_count": {
        "header": r"\textbf{SD3.5-Medium, counting}",
        "base": "countbase",
        "reward": "count",
        "rows": [
            (r"no anchor ($\beta=0$)   ", "cwb042", "cwb043"),
            (r"DDRL \; 0.0003          ", "cwd000342", "cwd000343"),
            (r"DDRL \; 0.001           ", "cwd00142", "cwd00143"),
            (r"DDRL \; 0.003           ", "cwd00342", "cwd00343"),
            (r"DDRL \; 0.01            ", "cwd0142", "cwd0143"),
            (r"FlowGRPO \; 0.01        ", "cwf00142", "cwf00143"),
            (r"FlowGRPO \; 0.04        ", "cwf00442", "cwf00443"),
            (r"FlowGRPO \; 0.1         ", "cwf0142", "cwf0143"),
        ],
    },
    "flux": {
        "header": r"\textbf{FLUX.1-dev, OCR}",
        "base": "fluxbase",
        "reward": "ocr",
        "rows": [
            (r"no anchor ($\beta=0$)   ", "fxb042", "fxb043"),
            (r"DDRL \; 0.001           ", "fxd00142", "fxd00143"),
            (r"FlowGRPO \; 0.04        ", "fxf00442", "fxf00443"),
        ],
    },
    # The repaired arms. Same configurations, same seeds, "r" suffix.
    "flux_fixed": {
        "header": r"\textbf{FLUX.1-dev, OCR}",
        "base": "fluxbase",
        "reward": "ocr",
        "rows": [
            (r"no anchor ($\beta=0$)   ", "fxb042r", "fxb043r"),
            (r"DDRL \; 0.001           ", "fxd00142r", "fxd00143r"),
            (r"FlowGRPO \; 0.04        ", "fxf00442r", "fxf00443r"),
        ],
    },
}


def load(tag):
    """The scored arm, preferring a plain final_eval_<tag>.json over the v2 one.

    The repaired arms are written without the v2 prefix, so new rows come from
    the current file while the published panels keep reading the archived ones.
    """
    for pat in (f"final_eval_{tag}.json", f"final_eval_v2_{tag}.json"):
        p = os.path.join(EVAL_DIR, pat)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    return None


def two_seed(a, b, key):
    """Mean over the seed pair and the standard error of that mean."""
    ma, mb = a[key]["mean"], b[key]["mean"]
    sa, sb = a[key]["sem"], b[key]["sem"]
    return (ma + mb) / 2.0, (sa ** 2 + sb ** 2) ** 0.5 / 2.0, abs(ma - mb)


def fmt(value, se, signed, places=3):
    """A cell, bold when the change clears two standard errors."""
    txt = f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
    if signed and abs(value) > 2 * se:
        return r"\bfseries " + txt
    return txt


def panel_rows(name):
    spec = PANELS[name]
    base = load(spec["base"])
    if base is None:
        raise SystemExit(f"no baseline json for {spec['base']}")
    out, missing = [], []
    task = spec["reward"]
    for label, t42, t43 in spec["rows"]:
        a, b = load(t42), load(t43)
        if a is None or b is None:
            missing.append(f"{label.strip()} ({t42}, {t43})")
            continue
        cells = []
        m, se, spread = two_seed(a, b, task)
        cells.append(fmt(m, se, signed=False))
        d_se = (se ** 2 + base[task]["sem"] ** 2) ** 0.5
        cells.append(fmt(m - base[task]["mean"], d_se, signed=True))
        for metric in METRICS:
            mm, mse, _ = two_seed(a, b, metric)
            dse = (mse ** 2 + base[metric]["sem"] ** 2) ** 0.5
            cells.append(fmt(mm - base[metric]["mean"], dse, signed=True))
        cells.append(f"{spread:.4f}")
        out.append((label, cells))
    head = [fmt(base[task]["mean"], 0.0, signed=False)] + [r"{--}"] * 5
    return spec, [("untrained               ", head)] + out, missing


def render(name):
    spec, rows, missing = panel_rows(name)
    lines = [rf"\multicolumn{{7}}{{l}}{{\textcolor{{hdr}}{{{spec['header']}}}}} \\",
             r"\addlinespace[2pt]"]
    for label, cells in rows:
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines), missing


# Pulls the body rows of a panel out of the .tex so --check can diff against
# what is actually published rather than against a copy of it.
def tex_rows(path, header_sub):
    body, grab = [], False
    for line in open(path):
        line = line.rstrip("\n")
        if r"\multicolumn{7}{l}" in line:
            grab = header_sub in line
            continue
        if grab:
            if r"\midrule" in line or r"\bottomrule" in line:
                grab = False
            elif line.strip().endswith(r"\\"):
                body.append(line)
    return body


def norm(line):
    """Compare on content, not on the column padding."""
    line = line.replace(r"\bfseries", " BOLD ")
    return re.sub(r"\s+", " ", line).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tex", default=os.path.join(os.path.dirname(__file__), "ddrl_grid.tex"))
    args = ap.parse_args()

    if args.check:
        bad = 0
        for name, header_sub in (("sd35_ocr", "SD3.5-Medium, OCR"),
                                 ("sd35_count", "SD3.5-Medium, counting"),
                                 ("flux", "FLUX.1-dev, OCR")):
            gen, _ = render(name)
            gen_rows = [l for l in gen.splitlines() if l.strip().endswith(r"\\")
                        and r"\multicolumn" not in l]
            pub = tex_rows(args.tex, header_sub)
            print(f"--- {name}: {len(gen_rows)} generated, {len(pub)} published")
            for g, p in zip(gen_rows, pub):
                if norm(g) != norm(p):
                    bad += 1
                    print(f"  MISMATCH\n    tex: {norm(p)}\n    gen: {norm(g)}")
            if len(gen_rows) != len(pub):
                bad += 1
                print("  row count differs")
        print("CHECK OK" if not bad else f"CHECK FAILED ({bad})")
        raise SystemExit(1 if bad else 0)

    for name in ([args.panel] if args.panel else list(PANELS)):
        out, missing = render(name)
        print(f"% {name}")
        print(out)
        if missing:
            print("% not scored yet: " + "; ".join(missing))
        print()


if __name__ == "__main__":
    main()
