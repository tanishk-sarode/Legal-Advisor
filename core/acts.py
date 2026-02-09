from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ActSource:
    act: str
    act_abbrev: str
    file_path: Path
    source_type: str


def get_act_sources(root: Path) -> List[ActSource]:
    acts_dir = root / "Indian-Law-Penal-Code-Json-main"
    return [
        ActSource(
            act="Indian Penal Code, 1860",
            act_abbrev="IPC",
            file_path=acts_dir / "ipc.json",
            source_type="section"
        ),
        ActSource(
            act="Code of Criminal Procedure, 1973",
            act_abbrev="CrPC",
            file_path=acts_dir / "crpc.json",
            source_type="section"
        ),
        ActSource(
            act="Civil Procedure Code, 1908",
            act_abbrev="CPC",
            file_path=acts_dir / "cpc.json",
            source_type="section"
        ),
        ActSource(
            act="Hindu Marriage Act, 1955",
            act_abbrev="HMA",
            file_path=acts_dir / "hma.json",
            source_type="section"
        ),
        ActSource(
            act="Indian Divorce Act, 1869",
            act_abbrev="IDA",
            file_path=acts_dir / "ida.json",
            source_type="section"
        ),
        ActSource(
            act="Indian Evidence Act, 1872",
            act_abbrev="IEA",
            file_path=acts_dir / "iea.json",
            source_type="section"
        ),
        ActSource(
            act="Negotiable Instruments Act, 1881",
            act_abbrev="NIA",
            file_path=acts_dir / "nia.json",
            source_type="section"
        ),
        ActSource(
            act="Motor Vehicles Act, 1988",
            act_abbrev="MVA",
            file_path=acts_dir / "MVA.json",
            source_type="section"
        )
    ]


def get_constitution_source(root: Path) -> ActSource:
    return ActSource(
        act="Constitution of India",
        act_abbrev="COI",
        file_path=root / "constitution_of_india.json",
        source_type="article"
    )
