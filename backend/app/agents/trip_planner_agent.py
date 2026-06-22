"""多智能体旅行规划系统"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from diskcache import Cache
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool
from pydantic import ValidationError
from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel, Budget
from ..config import get_settings

# ============ Agent提示词 ============

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

**重要提示:**
你必须使用工具来搜索景点!不要自己编造景点信息!

**工具调用格式:**
使用maps_text_search工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=景点关键词,city=城市名]`

**示例:**
用户: "搜索北京的历史文化景点"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=历史文化,city=北京]

用户: "搜索上海的公园"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=公园,city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 参数用逗号分隔
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

**重要提示:**
你必须使用工具来查询天气!不要自己编造天气信息!

**工具调用格式:**
使用maps_weather工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_weather:city=城市名]`

**示例:**
用户: "查询北京天气"
你的回复: [TOOL_CALL:amap_maps_weather:city=北京]

用户: "上海的天气怎么样"
你的回复: [TOOL_CALL:amap_maps_weather:city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。

**重要提示:**
你必须使用工具来搜索酒店!不要自己编造酒店信息!

**工具调用格式:**
使用maps_text_search工具搜索酒店时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=城市名]`

**示例:**
用户: "搜索北京的酒店"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=酒店,city=北京]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 关键词使用"酒店"或"宾馆"
"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""


# 缓存实例
_trip_cache = Cache("cache/trip_cache")


class MultiAgentTripPlanner:
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("开始初始化多智能体旅行规划系统...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            self.amap_tool = MCPTool(
                name="amap",
                description="高德地图服务",
                server_command=["uvx", "amap-mcp-server"],
                env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                auto_expand=True
            )
            self.amap_tool.expandable = True

            print("  - 创建景点搜索Agent...")
            self.attraction_agent = SimpleAgent(
                name="景点搜索专家",
                llm=self.llm,
                system_prompt=ATTRACTION_AGENT_PROMPT
            )
            self.attraction_agent.add_tool(self.amap_tool)

            print("  - 创建天气查询Agent...")
            self.weather_agent = SimpleAgent(
                name="天气查询专家",
                llm=self.llm,
                system_prompt=WEATHER_AGENT_PROMPT
            )
            self.weather_agent.add_tool(self.amap_tool)

            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = SimpleAgent(
                name="酒店推荐专家",
                llm=self.llm,
                system_prompt=HOTEL_AGENT_PROMPT
            )
            self.hotel_agent.add_tool(self.amap_tool)

            print("  - 创建行程规划Agent...")
            self.planner_agent = SimpleAgent(
                name="行程规划专家",
                llm=self.llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )

            print(f"[OK] 多智能体系统初始化成功")

        except Exception as e:
            print(f"[FAIL] 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # ──────────────────────────────────────────
    # 公开方法
    # ──────────────────────────────────────────

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """多智能体协作生成旅行计划，含缓存 + 自动重试与降级"""

        # --- 缓存检查 ---
        cache_key = self._cache_key(request)
        if cache_key in _trip_cache:
            print(f"  [CACHE] 命中行程缓存: {request.city} {request.travel_days}天")
            return _trip_cache[cache_key]

        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"\n[RETRY] 第 {attempt} 次重试...")

                print(f"\n目的地: {request.city} / {request.start_date} 至 {request.end_date} / {request.travel_days}天")

                # 步骤1-3: 搜索数据
                attraction_response = self._safe_run_agent(
                    self.attraction_agent,
                    self._build_attraction_query(request),
                    "景点搜索"
                )
                weather_response = self._safe_run_agent(
                    self.weather_agent,
                    f"请查询{request.city}的天气信息",
                    "天气查询"
                )
                hotel_response = self._safe_run_agent(
                    self.hotel_agent,
                    f"请搜索{request.city}的{request.accommodation}酒店",
                    "酒店搜索"
                )

                # 步骤4: 生成行程
                planner_query = self._build_planner_query(
                    request, attraction_response, weather_response, hotel_response
                )
                planner_response = self._safe_run_agent(
                    self.planner_agent, planner_query, "行程规划"
                )

                # 解析并校验
                trip_plan = self._parse_response(planner_response, request)

                # --- 写入缓存（24小时有效） ---
                _trip_cache.set(cache_key, trip_plan, expire=86400)
                print(f"  [CACHE] 已缓存: {request.city} {request.travel_days}天（24小时有效）")
                return trip_plan

            except Exception as e:
                print(f"[WARN] 第 {attempt+1} 次尝试失败: {str(e)[:100]}")
                if attempt >= max_retries:
                    print("[FALLBACK] 使用备用方案生成计划")
                    return self._create_fallback_plan(request)

        return self._create_fallback_plan(request)

    @staticmethod
    def _cache_key(request: TripRequest) -> str:
        """生成缓存 key：城市+天数+偏好+住宿+交通"""
        prefs = "|".join(request.preferences) if request.preferences else "无"
        extra = request.free_text_input or ""
        return f"trip:{request.city}|{request.travel_days}|{request.accommodation}|{request.transportation}|{prefs}|{extra}"

    # ──────────────────────────────────────────
    # Agent 安全执行
    # ──────────────────────────────────────────

    def _safe_run_agent(self, agent: SimpleAgent, query: str, label: str) -> str:
        """安全执行 Agent，捕获异常"""
        print(f"  [{label}] 正在执行...")
        try:
            response = agent.run(query)
            if not response or not response.strip():
                raise ValueError(f"{label} 返回空结果")
            return response
        except Exception as e:
            raise RuntimeError(f"{label} 失败: {str(e)}")

    # ──────────────────────────────────────────
    # 查询构建
    # ──────────────────────────────────────────

    def _build_attraction_query(self, request: TripRequest) -> str:
        keywords = request.preferences[0] if request.preferences else "景点"
        return (f"请使用amap_maps_text_search工具搜索{request.city}的{keywords}相关景点。\n"
                f"[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]")

    def _build_planner_query(self, request: TripRequest, attractions: str,
                              weather: str, hotels: str = "") -> str:
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息:**
{attractions}

**天气信息:**
{weather}

**酒店信息:**
{hotels}

**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店(从酒店信息中选择)
4. 考虑景点之间的距离和交通方式
5. 返回完整的JSON格式数据
6. 景点的经纬度坐标要真实准确
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"
        return query

    # ──────────────────────────────────────────
    # 响应解析（核心加固）
    # ──────────────────────────────────────────

    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        从 LLM 响应中提取 JSON，用 Pydantic 校验，失败抛异常触发重试
        """
        json_str = self._extract_json_str(response)
        if not json_str:
            raise ValueError("响应中未找到有效的JSON数据")

        data = self._robust_json_decode(json_str)
        if not data:
            raise ValueError("JSON 格式解析失败")

        data = self._fill_missing_fields(data, request)

        try:
            trip_plan = TripPlan(**data)
            if len(trip_plan.days) < request.travel_days:
                print(f"[WARN] 行程天数不足 (期望{request.travel_days}, 实际{len(trip_plan.days)})")
            return trip_plan
        except ValidationError as e:
            errors = e.errors()
            error_summary = "; ".join(f"{err['loc']}: {err['msg']}" for err in errors[:5])
            raise ValueError(f"数据校验失败: {error_summary}")

    @staticmethod
    def _extract_json_str(text: str) -> Optional[str]:
        """从 LLM 响应中健壮地提取 JSON 字符串"""
        if not text:
            return None

        for pattern in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```']:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()

        brace_depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    return text[start:i+1]

        return None

    @staticmethod
    def _robust_json_decode(json_str: str) -> Optional[dict]:
        """容错 JSON 解析，自动修复常见格式问题"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        fixed = re.sub(r'//.*?(\n|$)', '', fixed)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        fixed = fixed.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _fill_missing_fields(data: dict, request: TripRequest) -> dict:
        """补全 LLM 输出中可能缺失的字段"""
        data.setdefault("city", request.city)
        data.setdefault("start_date", request.start_date)
        data.setdefault("end_date", request.end_date)
        data.setdefault("overall_suggestions",
                        f"为您规划的{request.city}{request.travel_days}日游行程，祝你旅途愉快。")
        data.setdefault("weather_info", [])
        data.setdefault("budget", None)

        for day in data.get("days", []):
            day.setdefault("date", request.start_date)
            day.setdefault("day_index", 0)
            day.setdefault("description", "当日行程")
            day.setdefault("transportation", request.transportation)
            day.setdefault("accommodation", request.accommodation)
            day.setdefault("attractions", [])
            day.setdefault("meals", [])

        return data

    # ──────────────────────────────────────────
    # 降级方案
    # ──────────────────────────────────────────

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划"""
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")

        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            days.append(DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=116.4, latitude=39.9),
                        visit_duration=120,
                        description=f"{request.city}的著名景点",
                        category="景点",
                        ticket_price=0,
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"早餐", description="当地特色早餐", estimated_cost=20),
                    Meal(type="lunch", name=f"午餐", description="午餐推荐", estimated_cost=50),
                    Meal(type="dinner", name=f"晚餐", description="晚餐推荐", estimated_cost=80),
                ]
            ))

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。",
            budget=Budget(
                total_attractions=0,
                total_hotels=request.travel_days * 300,
                total_meals=request.travel_days * 150,
                total_transportation=request.travel_days * 50,
                total=request.travel_days * 500,
            )
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner
