#!/usr/bin/env python3
"""Prepare editable charts before the intentional manual rasterization step."""
from pathlib import Path
import argparse
import zipfile


PANEL_SD = [
    "0.042354428361323686",   # Pythia-70M
    "0.013941699604002449",   # SmolLM2-135M
    "0.18508700417530938",    # Pythia-160M
    "0.48628701602722196",    # Pythia-410M
    "0.012178416095887648",   # SmolLM2-360M
    "0.0982083669163503",     # GPT-Neo-125M
    "0.29827891586896504",    # OPT-125M
    "0.043213581647702955",   # GPT-2
    "0.039684777368568604",   # GPT-2-medium
    "0.228674491760278",      # BLOOM-560M
    "0.022765224286368756",   # OPT-350M
]


def number_literal(values):
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{value}</c:v></c:pt>'
        for index, value in enumerate(values)
    )
    return (
        '<c:numLit><c:formatCode>General</c:formatCode>'
        f'<c:ptCount val="{len(values)}"/>{points}</c:numLit>'
    )


def error_bars(values, color="142B3A"):
    return (
        "<c:errBars><c:errBarType val=\"both\"/>"
        '<c:errValType val="cust"/><c:noEndCap val="0"/>'
        f"<c:plus>{number_literal(values)}</c:plus>"
        f"<c:minus>{number_literal(values)}</c:minus>"
        f'<c:spPr><a:ln w="9525" cap="flat"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:round/></a:ln><a:effectLst/></c:spPr></c:errBars>'
    )


def point_color(index, color):
    return (
        f'<c:dPt><c:idx val="{index}"/><c:invertIfNegative val="0"/>'
        '<c:bubble3D val="0"/><c:spPr><a:solidFill>'
        f'<a:srgbClr val="{color}"/></a:solidFill><a:effectLst/></c:spPr></c:dPt>'
    )


def add_error_bars(xml, values, color="142B3A"):
    if "<c:errBars>" in xml or xml.count("<c:cat>") != 1:
        raise ValueError("chart XML is not in the expected unprepared form")
    return xml.replace("<c:cat>", error_bars(values, color) + "<c:cat>", 1)


def prepare(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")
    with zipfile.ZipFile(source) as archive:
        names = set(archive.namelist())
        required = {f"ppt/charts/chart{index}.xml" for index in (1, 2, 3, 4)}
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                "the preparation step requires the editable chart package; "
                f"missing {', '.join(missing)}"
            )
        entries = {name: archive.read(name) for name in names}

    entries["ppt/charts/chart1.xml"] = add_error_bars(
        entries["ppt/charts/chart1.xml"].decode(),
        ["0.004386929079948761", "0.019552720258314248"],
    ).encode()
    entries["ppt/charts/chart2.xml"] = add_error_bars(
        entries["ppt/charts/chart2.xml"].decode(),
        ["0.042354428361323686", "0.022765224286368756"],
    ).encode()
    chart4 = entries["ppt/charts/chart4.xml"].decode()
    if "<c:dPt>" in chart4:
        raise ValueError("chart4 already has a highlighted point")
    chart4 = chart4.replace(
        '<c:invertIfNegative val="0"/>',
        '<c:invertIfNegative val="0"/>' + point_color(0, "B5683C"),
        1,
    )
    entries["ppt/charts/chart4.xml"] = add_error_bars(
        chart4, PANEL_SD, "44606E"
    ).encode()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    print(f"prepared chart package: {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)


if __name__ == "__main__":
    main()
