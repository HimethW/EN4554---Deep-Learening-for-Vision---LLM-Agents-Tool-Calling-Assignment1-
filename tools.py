import ast
import operator
import re
import unicodedata

# made-up data so the model can't answer lookups from memory
WEATHER = {
    "london": (11, "light rain", 82),
    "paris": (14, "partly cloudy", 65),
    "tokyo": (19, "clear", 55),
    "new york": (8, "windy", 47),
    "sydney": (24, "sunny", 60),
    "colombo": (31, "thunderstorms", 88),
    "mumbai": (33, "hazy", 79),
    "berlin": (6, "overcast", 71),
    "cairo": (29, "sunny", 22),
    "toronto": (-3, "snow", 85),
    "oslo": (2, "foggy", 90),
    "dubai": (38, "clear", 35),
    "singapore": (30, "scattered showers", 84),
    "rome": (17, "sunny", 58),
    "moscow": (-8, "light snow", 76),
    "nairobi": (22, "cloudy", 63),
    "lima": (20, "drizzle", 81),
    "seoul": (9, "clear", 40),
    "kandy": (26, "light rain", 86),
    "reykjavik": (-1, "sleet", 77),
    "bangkok": (34, "humid", 74),
    "madrid": (15, "clear", 44),
}

STOCKS = {
    "AAPL": ("Apple", 214.37),
    "MSFT": ("Microsoft", 388.12),
    "GOOGL": ("Alphabet", 151.86),
    "AMZN": ("Amazon", 176.40),
    "TSLA": ("Tesla", 243.95),
    "NVDA": ("Nvidia", 131.25),
    "META": ("Meta", 502.18),
    "NFLX": ("Netflix", 612.70),
    "INTC": ("Intel", 31.44),
    "AMD": ("AMD", 159.03),
    "IBM": ("IBM", 197.61),
    "ORCL": ("Oracle", 142.80),
    "DIS": ("Disney", 98.57),
    "KO": ("Coca-Cola", 61.29),
    "NKE": ("Nike", 83.15),
}
COMPANY_TO_TICKER = {name.lower(): t for t, (name, _) in STOCKS.items()}

# (time, title, keywords accepted when checking answers)
EVENTS = {
    "monday": [("09:30", "Team standup", ["standup", "stand-up"]), ("14:00", "Dentist appointment", ["dentist"])],
    "tuesday": [("11:00", "Project review with Nimal", ["project review", "nimal"]), ("18:30", "Gym session", ["gym"])],
    "wednesday": [("10:00", "Robotics lab", ["lab", "robotics"]), ("16:00", "Call with supervisor", ["supervisor"])],
    "thursday": [("13:00", "Lunch with Sarah", ["lunch", "sarah"])],
    "friday": [("15:00", "Deep learning lecture", ["lecture", "deep learning"]), ("20:00", "Movie night", ["movie"])],
    "saturday": [("08:00", "Cricket practice", ["cricket"])],
    "sunday": [],
}
WEEKDAYS = list(EVENTS)

UNITS = {
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "mile": ("length", 1609.344), "foot": ("length", 0.3048), "inch": ("length", 0.0254), "yard": ("length", 0.9144),
    "kg": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.45359237), "oz": ("mass", 0.028349523),
    "l": ("volume", 1.0), "ml": ("volume", 0.001), "gallon": ("volume", 3.785411784), "cup": ("volume", 0.2365882),
    "celsius": ("temp", None), "fahrenheit": ("temp", None), "kelvin": ("temp", None),
}
UNIT_ALIASES = {
    "m": ["meter", "meters", "metre", "metres"],
    "km": ["kms", "kilometer", "kilometers", "kilometre", "kilometres"],
    "cm": ["centimeter", "centimeters", "centimetre", "centimetres"],
    "mm": ["millimeter", "millimeters", "millimetre", "millimetres"],
    "mile": ["mi", "miles"],
    "foot": ["ft", "feet"],
    "inch": ["in", "inches"],
    "yard": ["yd", "yds", "yards"],
    "kg": ["kgs", "kilo", "kilos", "kilogram", "kilograms"],
    "g": ["gram", "grams"],
    "lb": ["lbs", "pound", "pounds"],
    "oz": ["ounce", "ounces"],
    "l": ["liter", "liters", "litre", "litres"],
    "ml": ["milliliter", "milliliters", "millilitre", "millilitres"],
    "gallon": ["gal", "gallons"],
    "cup": ["cups"],
    "celsius": ["c", "°c", "centigrade"],
    "fahrenheit": ["f", "°f"],
    "kelvin": ["k"],
}
ALIAS_LOOKUP = {a: canon for canon, names in UNIT_ALIASES.items() for a in names + [canon]}


def canonical_unit(u):
    u = str(u).strip().lower().replace("degrees ", "").replace("degree ", "")
    return ALIAS_LOOKUP.get(u)


def canonical_city(c):
    c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode()
    return " ".join(c.lower().replace(",", " ").split())


def canonical_ticker(s):
    s = str(s).strip()
    if s.upper() in STOCKS:
        return s.upper()
    return COMPANY_TO_TICKER.get(s.lower())


def canonical_day(d):
    d = str(d).strip().lower()
    for day in WEEKDAYS:
        if d in (day, day[:3]):
            return day
    return None


def tidy(x):
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return round(x, 2) if abs(x) >= 1 else round(x, 4)


# safe arithmetic evaluation (no eval)
OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expr):
    expr = str(expr).replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    expr = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", " * ", expr)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return float(ev(ast.parse(expr, mode="eval")))


# the actual tools
def calculator(expression):
    try:
        return {"expression": expression, "result": tidy(safe_eval(expression))}
    except Exception:
        return {"error": f"could not evaluate '{expression}'"}


def convert_units(value, from_unit, to_unit):
    a, b = canonical_unit(from_unit), canonical_unit(to_unit)
    if a is None or b is None:
        return {"error": f"unknown unit '{from_unit if a is None else to_unit}'"}
    if UNITS[a][0] != UNITS[b][0]:
        return {"error": f"cannot convert {a} to {b}"}
    try:
        v = float(value)
    except (TypeError, ValueError):
        return {"error": f"value must be a number, got '{value}'"}
    if UNITS[a][0] == "temp":
        c = {"celsius": v, "fahrenheit": (v - 32) * 5 / 9, "kelvin": v - 273.15}[a]
        out = {"celsius": c, "fahrenheit": c * 9 / 5 + 32, "kelvin": c + 273.15}[b]
    else:
        out = v * UNITS[a][1] / UNITS[b][1]
    return {"value": tidy(v), "from_unit": a, "to_unit": b, "result": tidy(out)}


def get_weather(city):
    key = canonical_city(city)
    if key not in WEATHER:
        return {"error": f"no weather data for '{city}'"}
    t, cond, hum = WEATHER[key]
    return {"city": key.title(), "temperature_c": t, "condition": cond, "humidity_pct": hum}


def get_stock_price(symbol):
    t = canonical_ticker(symbol)
    if t is None:
        return {"error": f"unknown symbol '{symbol}'"}
    name, price = STOCKS[t]
    return {"symbol": t, "company": name, "price_usd": price}


def get_events(day):
    d = canonical_day(day)
    if d is None:
        return {"error": f"'{day}' is not a weekday name"}
    return {"day": d, "events": [{"time": tm, "title": title} for tm, title, _ in EVENTS[d]]}


FUNCTIONS = {
    "calculator": calculator,
    "convert_units": convert_units,
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "get_events": get_events,
}


def run_tool(name, args):
    if name not in FUNCTIONS:
        return {"error": f"unknown tool '{name}'"}
    try:
        return FUNCTIONS[name](**args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOL_SCHEMAS = [
    _fn("calculator", "Evaluate an arithmetic expression exactly. Supports + - * / ** % and parentheses.",
        {"expression": {"type": "string", "description": "e.g. '(12 + 7) * 3'"}}, ["expression"]),
    _fn("convert_units", "Convert a value between units of length, mass, volume or temperature.",
        {"value": {"type": "number"},
         "from_unit": {"type": "string", "description": "e.g. km, mile, kg, lb, celsius"},
         "to_unit": {"type": "string"}}, ["value", "from_unit", "to_unit"]),
    _fn("get_weather", "Get the current weather (temperature in celsius) for a city.",
        {"city": {"type": "string"}}, ["city"]),
    _fn("get_stock_price", "Get the latest stock price in USD for a ticker symbol.",
        {"symbol": {"type": "string", "description": "ticker, e.g. AAPL"}}, ["symbol"]),
    _fn("get_events", "List the user's calendar events for a day of the week.",
        {"day": {"type": "string", "description": "weekday name, e.g. monday"}}, ["day"]),
]


def same_call(name, args, exp_name, exp_args):
    # compares a call against the expected one, after normalising the arguments
    if name != exp_name or not isinstance(args, dict):
        return False
    try:
        if name == "calculator":
            return abs(safe_eval(args["expression"]) - safe_eval(exp_args["expression"])) < 1e-6
        if name == "convert_units":
            return (abs(float(args["value"]) - float(exp_args["value"])) < 1e-6
                    and canonical_unit(args["from_unit"]) == canonical_unit(exp_args["from_unit"])
                    and canonical_unit(args["to_unit"]) == canonical_unit(exp_args["to_unit"]))
        if name == "get_weather":
            return canonical_city(args["city"]) == canonical_city(exp_args["city"])
        if name == "get_stock_price":
            return canonical_ticker(args["symbol"]) == canonical_ticker(exp_args["symbol"])
        if name == "get_events":
            return canonical_day(args["day"]) == canonical_day(exp_args["day"])
    except Exception:
        return False
    return False
