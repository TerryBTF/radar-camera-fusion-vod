# Figure Sources

These figures are generated from Graphviz DOT sources.

- `ppt_simplified.dot`: simplified diagram for slides.
- `report_architecture.dot`: slightly more detailed diagram for reports.

Render commands:

```bash
dot -Tsvg ppt_simplified.dot -o ppt_simplified.svg
dot -Tpng ppt_simplified.dot -o ppt_simplified.png
dot -Tsvg report_architecture.dot -o report_architecture.svg
dot -Tpng report_architecture.dot -o report_architecture.png
```
