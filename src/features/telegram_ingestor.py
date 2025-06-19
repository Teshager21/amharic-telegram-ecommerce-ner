# src/features/telegram_ingestor.py
"""
Telegram Ingestor for Amharic E-Commerce Channels
Author: Teshager Admasu
Date: 2025-06-19

This module connects to Telegram using Telethon,
scrapes messages from specified channels,
normalizes Amharic text, and saves structured data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# from datetime import datetime

from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto

# from telethon.utils import pack_bot_file_id

import pandas as pd
import re
import unicodedata

from omegaconf import DictConfig, OmegaConf
import hydra

# Configure logging
logging.basicConfig(level=logging.INFO, format="📦 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TelegramIngestor:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "anon",
        channels: Optional[List[str]] = None,
        output_dir: Path = Path("data"),
    ):
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.channels = channels or []
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def connect(self):
        self.client.start()
        logger.info("🔌 Connected to Telegram.")

    def normalize_amharic(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u1369-\u137C]+", "", text)  # Remove Ethiopic numerals
        text = re.sub(r"[\u200B\u200C]", "", text)  # Remove zero-width chars
        text = re.sub(r"[\n\r]+", " ", text)  # Normalize newlines
        text = re.sub(r"[።፣፤፥፦፧]+", ".", text)  # Replace Amharic punctuations
        return text.strip()

    def scrape_channel(self, channel: str, limit: int = 1000) -> List[Dict]:
        logger.info(f"📥 Scraping channel: {channel}")
        messages = []
        media_dir = self.output_dir / "raw" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        for message in self.client.iter_messages(channel, limit=limit):
            if not message.text and not message.media:
                continue

            norm_text = self.normalize_amharic(message.text or "")
            has_photo = isinstance(message.media, MessageMediaPhoto)
            image_path = None

            if has_photo:
                try:
                    image_path = media_dir / f"{message.id}.jpg"
                    self.client.download_media(message, file=image_path)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to download image for message {message.id}: {e}"
                    )

            messages.append(
                {
                    "message_id": message.id,
                    "timestamp": message.date.isoformat(),
                    "sender": str(message.sender_id),
                    "text": norm_text,
                    "views": message.views or 0,
                    "has_photo": has_photo,
                    "image_path": str(image_path) if image_path else None,
                    "channel": channel,
                }
            )

        return messages

    def run(self, messages_limit: int = 1000):
        all_messages = []

        self.connect()
        for channel in self.channels:
            try:
                msgs = self.scrape_channel(channel, limit=messages_limit)
                all_messages.extend(msgs)
            except Exception as e:
                logger.error(f"❌ Failed scraping {channel}: {e}")

        # Save raw
        raw_path = self.output_dir / "raw" / "messages.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as f:
            for msg in all_messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        logger.info(f"✅ Saved raw messages to {raw_path}")

        # Save interim Parquet
        df = pd.DataFrame(all_messages)
        interim_path = self.output_dir / "interim" / "preprocessed.parquet"
        interim_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(interim_path, index=False)
            logger.info(f"✅ Saved preprocessed data to {interim_path}")
        except ImportError as e:
            logger.error(
                "❌ pyarrow or fastparquet is not installed. "
                "Please install one to save parquet files."
            )
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to save parquet file: {e}")
            raise e


@hydra.main(config_path="../../config", config_name="dev", version_base="1.3")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd

    project_root = Path(get_original_cwd())
    secret_path = project_root / "config" / "local.yaml"
    if secret_path.exists():
        secrets = OmegaConf.load(secret_path)
        cfg = OmegaConf.merge(cfg, secrets)
    else:
        logger.warning("⚠️ local.yaml not found; using dev.yaml only.")

    logger.info(f"📌 API ID: {cfg.telegram.api_id}")
    logger.info(f"📌 API HASH: {cfg.telegram.api_hash}")
    logger.info("🚀 Starting Telegram ingestion with Hydra config...")

    ingestor = TelegramIngestor(
        api_id=cfg.telegram.api_id,
        api_hash=cfg.telegram.api_hash,
        session_name=cfg.telegram.session_name,
        channels=cfg.telegram.channels,
        output_dir=Path(cfg.data.output_dir),
    )
    ingestor.run(messages_limit=cfg.telegram.limit)


if __name__ == "__main__":
    main()
