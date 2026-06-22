"""高德地图MCP服务封装"""

import json
import re
from typing import List, Dict, Any, Optional
from diskcache import Cache
from hello_agents.tools import MCPTool
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo

# 磁盘缓存（重启不丢）
_cache = Cache("cache/amap_cache")

# 全局MCP工具实例
_amap_mcp_tool = None


def get_amap_mcp_tool() -> MCPTool:
    """
    获取高德地图MCP工具实例(单例模式)
    
    Returns:
        MCPTool实例
    """
    global _amap_mcp_tool
    
    if _amap_mcp_tool is None:
        settings = get_settings()
        
        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
        
        # 创建MCP工具
        _amap_mcp_tool = MCPTool(
            name="amap",
            description="高德地图服务,支持POI搜索、路线规划、天气查询等功能",
            server_command=["uvx", "amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            auto_expand=True  # 自动展开为独立工具
        )
        
        print(f"[OK] 高德地图MCP工具初始化成功")
    
    return _amap_mcp_tool


def _cached_mcp_call(cache_key: str, tool_name: str, arguments: dict,
                     ttl: int = 86400) -> str:
    """
    带缓存的 MCP 工具调用。
    相同的 cache_key 在 ttl 秒内直接返回缓存结果。
    """
    key = f"mcp:{tool_name}:{cache_key}"
    if key in _cache:
        print(f"  [CACHE] 命中 {tool_name}({cache_key[:40]})")
        return _cache[key]

    print(f"  [MCP]  调用 {tool_name}({cache_key[:40]})...")
    tool = get_amap_mcp_tool()
    result = tool.run({
        "action": "call_tool",
        "tool_name": tool_name,
        "arguments": arguments
    })

    _cache.set(key, result, expire=ttl)
    return result


class AmapService:
    """高德地图服务封装类"""
    
    def __init__(self):
        """初始化服务"""
        self.mcp_tool = get_amap_mcp_tool()
    
    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI（带缓存）
        """
        try:
            cache_key = f"{keywords}|{city}"
            result = _cached_mcp_call(cache_key, "maps_text_search", {
                "keywords": keywords,
                "city": city,
                "citylimit": str(citylimit).lower()
            })
            # TODO: 解析实际的POI数据
            return []
        except Exception as e:
            print(f"[FAIL] POI搜索失败: {str(e)}")
            return []
    
    def get_weather(self, city: str) -> List[WeatherInfo]:
        """
        查询天气（带缓存）
        """
        try:
            result = _cached_mcp_call(city, "maps_weather", {"city": city})
            # TODO: 解析实际的天气数据
            return []
        except Exception as e:
            print(f"[FAIL] 天气查询失败: {str(e)}")
            return []
    
    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Dict[str, Any]:
        """
        规划路线（带缓存）
        """
        try:
            tool_map = {
                "walking": "maps_direction_walking_by_address",
                "driving": "maps_direction_driving_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }
            tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")
            
            arguments = {
                "origin_address": origin_address,
                "destination_address": destination_address
            }
            if origin_city:
                arguments["origin_city"] = origin_city
            if destination_city:
                arguments["destination_city"] = destination_city
            
            cache_key = f"{origin_address}|{destination_address}|{route_type}"
            result = _cached_mcp_call(cache_key, tool_name, arguments)
            return {}
        except Exception as e:
            print(f"[FAIL] 路线规划失败: {str(e)}")
            return {}
    
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """
        地理编码（带缓存）
        """
        try:
            arguments = {"address": address}
            if city:
                arguments["city"] = city

            cache_key = f"{address}|{city}" if city else address
            result = _cached_mcp_call(cache_key, "maps_geo", arguments)
            return None
        except Exception as e:
            print(f"[FAIL] 地理编码失败: {str(e)}")
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        获取POI详情（带缓存）
        """
        try:
            result = _cached_mcp_call(poi_id, "maps_search_detail", {"id": poi_id})

            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"raw": result}
        except Exception as e:
            print(f"[FAIL] 获取POI详情失败: {str(e)}")
            return {}


# 创建全局服务实例
_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service
    
    if _amap_service is None:
        _amap_service = AmapService()
    
    return _amap_service
