from src.config import AppConfig
from src.pipeline import CheatingDetectionPipeline


def main() -> None:
    config = AppConfig()
    pipeline = CheatingDetectionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()