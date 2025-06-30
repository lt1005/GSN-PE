import networkx as nx
import logging
from typing import Set, List, Tuple

class KGLoader:
    """知识图谱加载器"""
    
    def __init__(self):
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.triples: List[Tuple[str, str, str]] = []
    
    def load_kg(self, filepath: str) -> nx.MultiDiGraph:
        """加载知识图谱三元组"""
        G = nx.MultiDiGraph()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) != 3:
                        logging.warning(f"第{line_num}行格式错误: {line}")
                        continue
                    
                    h, r, t = parts
                    self.entities.update([h, t])
                    self.relations.add(r)
                    self.triples.append((h, r, t))
                    G.add_edge(h, t, relation=r)
            
            logging.info(f"成功加载知识图谱: {len(self.entities)}个实体, {len(self.relations)}种关系, {len(self.triples)}个三元组")
        except FileNotFoundError:
            logging.error(f"文件未找到: {filepath}")
            raise
        except Exception as e:
            logging.error(f"加载知识图谱时出错: {e}")
            raise
        
        return G