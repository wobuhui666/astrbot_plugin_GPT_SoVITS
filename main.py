import asyncio
import re
import random
import aiohttp # 确保 aiohttp 已经安装 (通常AstrBot环境自带)
from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Record
from astrbot.core.platform import AstrMessageEvent
import astrbot.core.message.components as Comp
from pathlib import Path
from typing import Dict, Any

# --- (文件路径配置部分保持不变) ---
SAVED_AUDIO_DIR = Path("./data/plugins_data/astrbot_plugin_GPT_SoVITS")
REFERENCE_AUDIO_DIR: Path = (Path(__file__).resolve().parent / "reference_audio")
SAVED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@register(
    "astrbot_plugin_GPT_SoVITS",
    "Zhalslar",
    "GPT_SoVITS对接插件",
    "1.3.0", # 版本号更新，表示这是一个重要修复
    "https://github.com/Zhalslar/astrbot_plugin_GPT_SoVITS",
)
class GPTSoVITSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        # ( __init__ 函数的前半部分保持不变 )
        super().__init__(context)
        base_setting: Dict = config.get("base_setting", {})
        self.base_url: str = base_setting.get("base_url", "")
        auto_config: Dict = config.get("auto_config", {})
        self.send_record_probability: float = auto_config.get("send_record_probability", 0.15)
        self.max_resp_text_len: int = auto_config.get("max_resp_text_len", 50)
        role_config: Dict = config.get("role", {})
        self.default_emotion: str = role_config.get("default_emotion", "生气地")
        self.gpt_weights_path: str = role_config.get("gpt_weights_path", "")
        self.sovits_weights_path: str = role_config.get("sovits_weights_path", "")
        asyncio.create_task(self._set_model_weights())
        emotions_config = config.get("emotions", {})
        gently_config = emotions_config.get("gently", {})
        happily_config = emotions_config.get("happily", {})
        angrily_config = emotions_config.get("angrily", {})
        surprise_config = emotions_config.get("surprise", {})
        self.preset_emotions: Dict = {
            "温柔地说": {"ref_audio_path": gently_config.get("ref_audio_path") or str(REFERENCE_AUDIO_DIR / "不要害怕，也不要哭了.wav"),"prompt_text": gently_config.get("prompt_text") or "不要害怕，也不要哭了","prompt_lang": gently_config.get("prompt_lang"),"speed_factor": gently_config.get("speed_factor"),"fragment_interval": gently_config.get("fragment_interval"),},
            "开心地说": {"ref_audio_path": happily_config.get("ref_audio_path") or str(REFERENCE_AUDIO_DIR / "它好像在等另一只蕈兽_心情很好的样子.wav"),"prompt_text": happily_config.get("prompt_text") or "它好像在等另一只蕈兽_心情很好的样子","prompt_lang": happily_config.get("prompt_lang"),"speed_factor": happily_config.get("speed_factor"),"fragment_interval": happily_config.get("fragment_interval"),},
            "生气地说": {"ref_audio_path": angrily_config.get("ref_audio_path") or str(REFERENCE_AUDIO_DIR/ "你还会选择现在的位置吗？到那时，你觉得自己又会是什么呢.wav"),"prompt_text": angrily_config.get("prompt_text") or "你还会选择现在的位置吗？到那时，你觉得自己又会是什么呢","prompt_lang": angrily_config.get("prompt_lang"),"speed_factor": angrily_config.get("speed_factor"),"fragment_interval": angrily_config.get("fragment_interval"),},
            "惊讶地说": {"ref_audio_path": surprise_config.get("ref_audio_path") or str(REFERENCE_AUDIO_DIR / "就算是这样，也不至于直接碎掉啊，除非.wav"),"prompt_text": surprise_config.get("prompt_text") or "就算是这样，也不至于直接碎掉啊，除非","prompt_lang": surprise_config.get("prompt_lang"),"speed_factor": surprise_config.get("speed_factor"),"fragment_interval": surprise_config.get("fragment_interval"),},
        }
        self.preset_emotions_set = set(self.preset_emotions.keys())
        self.keywords_dict = {"温柔地说": gently_config.get("keywords"),"开心地说": happily_config.get("keywords"),"生气地说": angrily_config.get("keywords"),"惊讶地说": surprise_config.get("keywords"),}
        self.default_params: Dict = config.get("default_params", {})

    # _make_request 函数保持原样，用于其他GET请求
    async def _make_request(
        self,
        endpoint: str,
        params=None,
    ) -> None | bytes:
        if params:
            params = {
                k: str(v).lower() if isinstance(v, bool) else v
                for k, v in params.items()
            }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request("GET", endpoint, params=params) as response:
                    response.raise_for_status()
                    audio_bytes = await response.read()
                    return audio_bytes
            except aiohttp.ClientError as e:
                logger.error(f"Request to {endpoint} failed: {e}")
                return None

    async def _set_model_weights(self):
        # (此函数不变，它正确地使用了_make_request发送GET请求)
        try:
            if self.gpt_weights_path:
                gpt_endpoint = f"{self.base_url}/set_gpt_weights"
                gpt_params = {"weights_path": self.gpt_weights_path}
                if await self._make_request(endpoint=gpt_endpoint, params=gpt_params): logger.info(f"成功设置 GPT 模型路径：{self.gpt_weights_path}")
            else: logger.info("GPT 模型路径未配置，将使用GPT_SoVITS内置的GPT模型")
            if self.sovits_weights_path:
                sovits_endpoint = f"{self.base_url}/set_sovits_weights"
                sovits_params = {"weights_path": self.sovits_weights_path}
                if await self._make_request(endpoint=sovits_endpoint, params=sovits_params): logger.info(f"成功设置 SoVITS 模型路径：{self.sovits_weights_path}")
            else: logger.info("SoVITS 模型路径未配置，将使用GPT_SoVITS内置的SoVITS模型")
        except aiohttp.ClientError as e: logger.error(f"设置模型路径时发生错误：{e}")
        except Exception as e: logger.error(f"发生未知错误：{e}")

    # (on_decorating_result 和 on_command 保持不变)
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        if random.random() > self.send_record_probability: return
        chain = event.get_result().chain; seg = chain[0]
        if not (len(chain) == 1 and isinstance(seg, Comp.Plain)): return
        resp_text = seg.text
        if len(resp_text) > self.max_resp_text_len: return
        send_text = event.message_str
        emotion = self.default_emotion
        for emo, keywords in self.keywords_dict.items():
            for keyword in keywords:
                if keyword in send_text or keyword in resp_text: emotion = emo; break
            else: continue
            break
        params = self.default_params.copy()
        params.update(self.preset_emotions[emotion])
        params["text"] = resp_text
        file_name = self.generate_file_name(event, params=params)
        save_path = await self.tts_inference(params=params, file_name=file_name)
        if save_path is None: logger.error("TTS任务执行失败！"); return
        chain.clear(); chain.append(Record.fromFileSystem(save_path))

    @filter.command("说", alias={"温柔地说", "开心地说", "生气地说", "惊讶地说",},)
    async def on_command(self, event: AstrMessageEvent, send_text: str | int | None = None):
        if not send_text: yield event.plain_result("未提供文本"); return
        send_text = str(send_text)
        emotion = next((emo for emo in self.preset_emotions_set if emo in event.get_message_str()), self.default_emotion,)
        params = self.default_params.copy()
        params.update(self.preset_emotions[emotion])
        params["text"] = send_text
        if not emotion or not send_text: return
        file_name = self.generate_file_name(event, params=params)
        save_path = await self.tts_inference(params=params, file_name=file_name)
        if save_path is None: logger.error("TTS任务执行失败！"); return
        chain = [Record.fromFileSystem(save_path)]
        yield event.chain_result(chain)

    def generate_file_name(self, event: AstrMessageEvent, params) -> str:
        # (此函数不变)
        group_id = event.get_group_id() or "0"; sender_id = event.get_sender_id() or "0"
        sanitized_text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s]", "", params["text"])
        limit_text = sanitized_text.strip()[:30]
        media_type = self.default_params["media_type"]
        file_name = f"{group_id}_{sender_id}_{limit_text}.{media_type}"
        return file_name

    # --- START OF MODIFIED CODE ---

    async def tts_inference(self, params: Dict[str, Any], file_name: str) -> str | None:
        """
        发送TTS请求，获取音频内容。
        此函数现在直接、显式地使用 POST 方法发送 JSON 请求体。
        """
        endpoint = f"{self.base_url}/tts"
        save_path = str((SAVED_AUDIO_DIR / file_name).resolve())
        
        # 为了健壮性，我们对布尔值进行预处理
        json_payload = {
            k: str(v).lower() if isinstance(v, bool) else v
            for k, v in params.items()
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            # 使用 aiohttp.ClientSession 直接发送一个明确的 POST 请求
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(
                    endpoint, 
                    json=json_payload,  # 使用 json 关键字参数来发送 JSON 请求体
                    timeout=180
                ) as response:
                    # 检查响应状态码，如果不是2xx，则会引发异常
                    response.raise_for_status() 
                    
                    audio_bytes = await response.read()
                    
                    if audio_bytes:
                        with open(save_path, "wb") as audio_file:
                            audio_file.write(audio_bytes)
                        return save_path
                    else:
                        logger.warning("TTS request successful, but received empty audio content.")
                        return None

        except aiohttp.ClientError as e:
            # 捕获所有 aiohttp 相关的错误 (网络错误, HTTP 4xx/5xx 错误等)
            logger.error(f"TTS request to {endpoint} failed: {e}")
            # 尝试读取并记录错误响应体，这对于调试非常有用
            if 'response' in locals() and hasattr(response, 'text'):
                error_body = await response.text()
                logger.error(f"Server error response body: {error_body}")
            return None
        except Exception as e:
            # 捕获其他未知错误
            logger.error(f"An unexpected error occurred during TTS inference: {e}")
            return None

    # --- END OF MODIFIED CODE ---

    @filter.command("重启TTS", alias={"重启tts"})
    async def tts_control(self, event: AstrMessageEvent):
        # (此函数不变)
        yield event.plain_result("重启TTS中...(报错信息请忽略，等待一会即可完成重启)")
        endpoint = f"{self.base_url}/control"
        params = {"command": "restart"}
        await self._make_request(endpoint=endpoint, params=params)

    async def tts_sever(self, text: str, file_name: str) -> str | None:
        # (此函数不变)
        emotion = self.default_emotion
        for emo, keywords in self.keywords_dict.items():
            for keyword in keywords:
                if keyword in text:
                    emotion = emo
                    break
        params = self.default_params.copy()
        params.update(self.preset_emotions[emotion])
        params["text"] = text
        save_path = await self.tts_inference(params=params, file_name=file_name)
        return save_path
