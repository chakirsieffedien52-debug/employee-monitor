# Third-party software and model licensing

Employee Monitor's own source code is offered under the MIT License in [LICENSE](LICENSE). That license does not replace the terms that apply to third-party packages, model weights, or other bundled materials. Review the applicable license texts and seek legal advice for a specific deployment.

## Direct runtime dependencies

| Component | Purpose | License |
| --- | --- | --- |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO detection and tracking | AGPL-3.0 |
| [OpenCV](https://github.com/opencv/opencv-python) | Video capture, image processing, and preview | Apache-2.0 |
| [PyYAML](https://github.com/yaml/pyyaml) | YAML configuration | MIT |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Optional local `.env` loading | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | Image and array operations | BSD-3-Clause |
| [FastAPI](https://github.com/fastapi/fastapi) | Optional dashboard API | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | Optional dashboard server | BSD-3-Clause |
| [ReportLab](https://github.com/MrBitBucket/reportlab) | PDF reports | BSD-style license; consult the package license |

Development dependencies are pytest (MIT), Ruff (MIT), Black (MIT), and pre-commit (MIT). Dependency versions and transitive dependencies are resolved by the package installer; inspect their metadata and license notices for the exact installed versions.

## YOLO package and weights

Detection uses pretrained YOLOv8 models through the Ultralytics package. Ultralytics identifies its open-source software as **AGPL-3.0** and offers a separate Enterprise license for certain uses. The package downloads model weights on first use; this repository does not include or redistribute the weight files. The applicable terms for the software, weights, modifications, and the way they are deployed must be assessed separately.

Ultralytics describes AGPL use as suitable when you are comfortable open-sourcing the full project and offers an Enterprise license for some private, proprietary, commercial, embedded, or network-service deployments. Read [Ultralytics' licensing information](https://www.ultralytics.com/license) and the [AGPL-3.0 license text](https://www.gnu.org/licenses/agpl-3.0.html) before redistribution, commercial use, private/internal deployment, or use in a network service. Do not assume that this repository's MIT license grants rights to Ultralytics software or models.

## Notices and source

No third-party source code or model weights are intentionally vendored in this repository. Where a distribution or deployment includes third-party components, include the notices and license texts required by those components' terms.
