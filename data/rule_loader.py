import json
import logging
from typing import List, Dict

class RuleLoader:
    """规则结构加载器"""
    
    def load_rules(self, filepath: str) -> List[Dict]:
        """加载规则结构"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            logging.info(f"成功加载{len(rules)}条规则")
            return rules
        except FileNotFoundError:
            logging.error(f"规则文件未找到: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误: {e}")
            raise