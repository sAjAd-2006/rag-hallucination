# HTML report sources

The two PDF reports in the submission are generated from the standalone HTML files in this folder.

To reproduce the layout in a browser:

1. Open the relevant `.html` file.
2. Print to PDF with paper size `A4`.
3. Use `100%` scale and `None` margins.
4. Enable background graphics.

The delivered PDFs were rendered with WeasyPrint 62.3 and pydyf 0.10.0. All report CSS is embedded inside each HTML file, so there is no separate stylesheet or web dependency. The visual system is adapted from the supplied `T05_Report(1).html` reference: a navy gradient cover, blue/green accents, section tags, white metric cards, and dark table headers.
