def get_model_id(model: str) -> str:
    # Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
    base_model_ids = {
        "claude-v2": "anthropic.claude-v2:1",
        "claude-instant-v1": "anthropic.claude-instant-v1",
        "claude-v3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-v3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-v3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-v3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude-v3.5-sonnet-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-v3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "mistral-7b-instruct": "mistral.mistral-7b-instruct-v0:2",
        "mixtral-8x7b-instruct": "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral-large": "mistral.mistral-large-2402-v1:0",
        # New Amazon Nova models
        "amazon-nova-pro": "amazon.nova-pro-v1:0",
        "amazon-nova-lite": "amazon.nova-lite-v1:0",
        "amazon-nova-micro": "amazon.nova-micro-v1:0",
    }

    base_model_id = base_model_ids.get(model)
    if not base_model_id:
        raise ValueError(f"Unsupported model: {model}")

    return base_model_id

US_TO_JPY_RATE = 150
BEDROCK_PRICING = {
    "us-east-1": {
        "claude-instant-v1": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v2": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-v3.5-haiku": {"input": 0.001, "output": 0.005},
        "claude-v3-sonnet": {"input": 0.00300, "output": 0.01500},
        "claude-v3.5-sonnet": {"input": 0.00300, "output": 0.01500},
        "claude-v3.5-sonnet-v2": {"input": 0.00300, "output": 0.01500},
        "mistral-7b-instruct": {"input": 0.00015, "output": 0.0002},
        "mixtral-8x7b-instruct": {"input": 0.00045, "output": 0.0007},
        "mistral-large": {"input": 0.008, "output": 0.024},
        "amazon-nova-pro": {"input": 0.0008, "output": 0.0032},
        "amazon-nova-lite": {"input": 0.00006, "output": 0.00024},
        "amazon-nova-micro": {"input": 0.000035, "output": 0.00014},
    },
    "us-west-2": {
        "claude-instant-v1": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v2": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v3-sonnet": {"input": 0.00300, "output": 0.01500},
        "claude-v3-opus": {"input": 0.01500, "output": 0.07500},
        "mistral-7b-instruct": {"input": 0.00015, "output": 0.0002},
        "mixtral-8x7b-instruct": {"input": 0.00045, "output": 0.0007},
        "mistral-large": {"input": 0.008, "output": 0.024},
        "amazon-nova-pro": {"input": 0.0008, "output": 0.0032},
        "amazon-nova-lite": {"input": 0.00006, "output": 0.00024},
        "amazon-nova-micro": {"input": 0.000035, "output": 0.00014},
    },
    "ap-northeast-1": {
        "claude-instant-v1": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v2": {
            "input": 0.00080,
            "output": 0.00240,
        },
    },
    "default": {
        "claude-instant-v1": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v2": {
            "input": 0.00080,
            "output": 0.00240,
        },
        "claude-v3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-v3.5-haiku": {"input": 0.001, "output": 0.005},
        "claude-v3-sonnet": {"input": 0.00300, "output": 0.01500},
        "claude-v3.5-sonnet": {"input": 0.00300, "output": 0.01500},
        "claude-v3.5-sonnet-v2": {"input": 0.00300, "output": 0.01500},
        "claude-v3-opus": {"input": 0.01500, "output": 0.07500},
        "mistral-7b-instruct": {"input": 0.00015, "output": 0.0002},
        "mixtral-8x7b-instruct": {"input": 0.00045, "output": 0.0007},
        "mistral-large": {"input": 0.008, "output": 0.024},
        "amazon-nova-pro": {"input": 0.0008, "output": 0.0032},
        "amazon-nova-lite": {"input": 0.00006, "output": 0.00024},
        "amazon-nova-micro": {"input": 0.000035, "output": 0.00014},
    },
}
