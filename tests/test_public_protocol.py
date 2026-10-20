import json
from pathlib import Path

import pytest

from experiments.evaluate_qwen3vl_recent4_full import validate_processor_contract
from scripts.configure_evaluation_protocol import configure
from viss_core.evaluation_protocol import ProtocolValidationError, load_locked_protocol


TEMPLATE = Path(__file__).resolve().parents[1] / "configs/evaluation/recent4.example.json"


def test_unconfigured_template_cannot_be_used_as_a_locked_protocol():
    with pytest.raises(ProtocolValidationError, match="not locked"):
        load_locked_protocol(TEMPLATE)


def test_user_configured_protocol_preserves_processor_identity_check(tmp_path):
    template = json.loads(TEMPLATE.read_text())
    contract = template["inference"]["processor_contract"]
    processor = tmp_path / "processor"
    processor.mkdir()
    config = processor / "preprocessor_config.json"
    config.write_text(json.dumps({
        key: contract[key]
        for key in ["image_processor_type", "patch_size", "merge_size", "temporal_patch_size"]
    } | {"size": {
        key: contract[key] for key in ["shortest_edge", "longest_edge"]
    }}))
    streaming = tmp_path / "streaming.jsonl"
    ovo = tmp_path / "ovo.jsonl"
    for path in (streaming, ovo):
        path.write_text('{"uid": "synthetic-query"}\n')
    output = tmp_path / "configured.json"
    configure(
        template=TEMPLATE,
        streaming_samples=streaming,
        ovo_samples=ovo,
        processor_config=config,
        output=output,
    )
    protocol = load_locked_protocol(output)
    validate_processor_contract(processor, protocol)
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256 diverged"):
        validate_processor_contract(processor, protocol)
