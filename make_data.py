import json
import os
import random

from tools import EVENTS, STOCKS, WEATHER, run_tool

SEED = 42
N_TRAIN = {"calc": 150, "convert": 150, "weather": 140, "stock": 120, "events": 110,
           "weather_convert": 45, "stock_calc": 35, "parallel": 50, "none": 200}
N_VAL = 100
N_TEST = {"calc": 14, "convert": 14, "weather": 14, "stock": 12, "events": 12,
          "weather_convert": 4, "stock_calc": 4, "parallel": 6, "none": 20}

# a few cities/tickers only appear in the test set
TEST_ONLY_CITIES = ["kandy", "reykjavik", "lima", "nairobi"]
TEST_ONLY_TICKERS = ["ORCL", "KO", "NKE"]


def call(name, **args):
    return {"name": name, "arguments": args, "result": run_tool(name, args)}


def city_name(c):
    return c.title()


def money(x):
    return f"{x:.2f}"


# ---------------- calculator ----------------
CALC = {
    "mul": {
        "make": lambda r: {"a": r.randint(12, 999), "b": r.randint(12, 99)},
        "expr": "{a} * {b}",
        "train": ["What is {a} times {b}?", "Calculate {a} * {b}.", "How much is {a} multiplied by {b}?",
                  "If I have {a} boxes with {b} items each, how many items is that?"],
        "test": ["multiply {a} by {b} for me", "{a}*{b} = ?",
                 "A shop sells {b} packs of {a} pencils. How many pencils in total?"],
    },
    "add": {
        "make": lambda r: {"a": r.randint(100, 9999), "b": r.randint(100, 9999), "c": r.randint(100, 9999)},
        "expr": "{a} + {b} + {c}",
        "train": ["What is {a} + {b} + {c}?", "Add {a}, {b} and {c}.", "What's the total of {a}, {b} and {c}?"],
        "test": ["I need the sum of {a}, {b} and {c}", "{a} plus {b} plus {c} equals what?"],
    },
    "sub": {
        "make": lambda r: (lambda b: {"a": b + r.randint(50, 9000), "b": b})(r.randint(100, 5000)),
        "expr": "{a} - {b}",
        "train": ["What is {a} minus {b}?", "Calculate {a} - {b}.", "How much is {a} take away {b}?"],
        "test": ["Subtract {b} from {a}.", "take {b} away from {a}, what's left?"],
    },
    "div": {
        "make": lambda r: {"a": r.randint(100, 20000), "b": r.randint(3, 48)},
        "expr": "{a} / {b}",
        "train": ["What is {a} divided by {b}?", "Calculate {a} / {b}.",
                  "Split a bill of {a} between {b} people. How much does each pay?"],
        "test": ["Quick math: {a}/{b}", "If {a} rupees is shared equally by {b} friends, how much does each get?"],
    },
    "pct": {
        "make": lambda r: {"p": r.choice([5, 8, 12, 15, 18, 20, 25, 35, 40, 65, 75]), "n": r.randint(40, 5000)},
        "expr": "{p} / 100 * {n}",
        "train": ["What is {p}% of {n}?", "Calculate {p} percent of {n}.", "How much is {p}% of {n}?"],
        "test": ["Could you work out {p} percent of {n}?", "how much is a {p}% tip on a {n} dollar bill"],
    },
    "paren": {
        "make": lambda r: {"a": r.randint(10, 500), "b": r.randint(10, 500), "c": r.randint(3, 60)},
        "expr": "({a} + {b}) * {c}",
        "train": ["Compute ({a} + {b}) * {c}.", "What is ({a} + {b}) times {c}?"],
        "test": ["Add {a} and {b}, then multiply the result by {c}."],
    },
    "pow": {
        "make": lambda r: {"a": r.randint(11, 60), "b": r.choice([2, 3])},
        "expr": "{a} ** {b}",
        "train": ["What is {a} to the power of {b}?", "Calculate {a}^{b}."],
        "test": ["What do you get when you raise {a} to power {b}?"],
    },
}
CALC_ANSWERS = ["The answer is {res}.", "{expr} = {res}.", "That works out to {res}."]


def gen_calc(r, split):
    task = CALC[r.choice(list(CALC))]
    f = task["make"](r)
    expr = task["expr"].format(**f)
    c = call("calculator", expression=expr)
    res = c["result"]["result"]
    q = r.choice(task[split]).format(**f)
    return q, [[c]], r.choice(CALC_ANSWERS).format(expr=expr, res=res), {"numbers": [res]}


# ---------------- unit conversion ----------------
PAIRS = [("km", "mile", "length"), ("mile", "km", "length"), ("m", "foot", "length"), ("foot", "m", "length"),
         ("inch", "cm", "length"), ("cm", "inch", "length"), ("yard", "m", "length"),
         ("kg", "lb", "mass"), ("lb", "kg", "mass"), ("g", "oz", "mass"), ("oz", "g", "mass"),
         ("l", "gallon", "volume"), ("gallon", "l", "volume"), ("cup", "ml", "volume"),
         ("celsius", "fahrenheit", "temp"), ("fahrenheit", "celsius", "temp"), ("celsius", "kelvin", "temp")]
RANGES = {"km": (1, 500), "mile": (1, 300), "m": (1, 500), "foot": (1, 1000), "inch": (1, 200), "cm": (1, 300),
          "yard": (1, 400), "kg": (1, 150), "lb": (1, 300), "g": (10, 1000), "oz": (1, 50), "l": (1, 60),
          "gallon": (1, 40), "cup": (1, 10), "celsius": (-20, 45), "fahrenheit": (0, 110)}
SURFACE = {"km": ["km", "kilometers", "kilometres"], "mile": ["miles"], "m": ["meters", "m", "metres"],
           "foot": ["feet", "ft"], "inch": ["inches"], "cm": ["cm", "centimeters"], "yard": ["yards"],
           "kg": ["kg", "kilograms", "kilos"], "lb": ["pounds", "lbs"], "g": ["grams", "g"],
           "oz": ["ounces", "oz"], "l": ["liters", "litres"], "gallon": ["gallons"], "cup": ["cups"],
           "ml": ["ml", "milliliters"], "celsius": ["celsius", "°C", "degrees Celsius"],
           "fahrenheit": ["fahrenheit", "°F", "degrees Fahrenheit"], "kelvin": ["kelvin"]}
CONVERT_TRAIN = ["Convert {v} {u1} to {u2}.", "How many {u2} is {v} {u1}?", "What is {v} {u1} in {u2}?",
                 "{v} {u1} is how many {u2}?"]
CONVERT_TEST = {None: ["Turn {v} {u1} into {u2}", "need {v} {u1} expressed in {u2}", "what would {v} {u1} be in {u2}?"],
                "length": ["I ran {v} {u1} today. What's that in {u2}?"],
                "mass": ["my suitcase weighs {v} {u1}, how much is that in {u2}"],
                "volume": ["The recipe needs {v} {u1}. How much is that in {u2}?"],
                "temp": ["It's {v} {u1} outside, what's that in {u2}?"]}


def gen_convert(r, split):
    a, b, cat = r.choice(PAIRS)
    lo, hi = RANGES[a]
    v = r.randint(lo, hi) if r.random() < 0.7 or cat == "temp" else round(r.uniform(lo, hi), 1)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    c = call("convert_units", value=v, from_unit=a, to_unit=b)
    u1, u2 = r.choice(SURFACE[a]), r.choice(SURFACE[b])
    pool = CONVERT_TRAIN if split == "train" else CONVERT_TEST[None] + CONVERT_TEST[cat]
    q = r.choice(pool).format(v=v, u1=u1, u2=u2)
    res = c["result"]["result"]
    return q, [[c]], f"{v} {u1} is about {res} {u2}.", {"numbers": [res]}


# ---------------- weather ----------------
WEATHER_TRAIN = ["What's the weather in {city}?", "How is the weather in {city} right now?",
                 "Tell me the current weather in {city}.", "What is the temperature in {city}?",
                 "Is it raining in {city}?", "Weather update for {city}, please."]
WEATHER_TEST = ["I'm flying to {city} tonight, what are the conditions like there?", "{city} weather?",
                "how hot or cold is it in {city} at the moment", "Do I need an umbrella in {city} today?",
                "Should I pack a jacket for {city}? What's it like there now?"]
WEATHER_ANSWERS = ["It's {t}°C and {cond} in {city}, with {hum}% humidity.",
                   "Right now {city} is {cond} at {t}°C (humidity {hum}%)."]


def pick_city(r, split):
    pool = [c for c in WEATHER if c not in TEST_ONLY_CITIES] if split == "train" else list(WEATHER)
    return r.choice(pool)


def show_city(r, c, split):
    return c if split == "test" and r.random() < 0.3 else city_name(c)


def gen_weather(r, split):
    city = pick_city(r, split)
    c = call("get_weather", city=city_name(city))
    w = c["result"]
    q = r.choice(WEATHER_TRAIN if split == "train" else WEATHER_TEST).format(city=show_city(r, city, split))
    ans = r.choice(WEATHER_ANSWERS).format(t=w["temperature_c"], cond=w["condition"], city=w["city"],
                                           hum=w["humidity_pct"])
    return q, [[c]], ans, {"numbers": [w["temperature_c"]]}


# ---------------- stocks ----------------
STOCK_TRAIN = ["What's the stock price of {t}?", "How much is {t} trading at?",
               "Give me the current price of {t} shares.", "What is {name}'s share price?", "Price check on {t} stock."]
STOCK_TEST = ["where is {t} trading right now", "quote for {t} please",
              "How much would one share of {name} cost me today?", "Latest {name} stock price?"]


def pick_ticker(r, split):
    pool = [t for t in STOCKS if t not in TEST_ONLY_TICKERS] if split == "train" else list(STOCKS)
    return r.choice(pool)


def gen_stock(r, split):
    t = pick_ticker(r, split)
    c = call("get_stock_price", symbol=t)
    s = c["result"]
    q = r.choice(STOCK_TRAIN if split == "train" else STOCK_TEST).format(t=t, name=s["company"])
    ans = r.choice(["{name} ({t}) is trading at ${p}.", "The latest price for {t} is ${p}."]).format(
        name=s["company"], t=t, p=money(s["price_usd"]))
    return q, [[c]], ans, {"numbers": [s["price_usd"]]}


# ---------------- calendar ----------------
EVENTS_TRAIN = ["What do I have on {day}?", "What's on my calendar for {day}?", "Do I have any meetings on {day}?",
                "Show me my schedule for {day}."]
EVENTS_TEST = ["am I free on {day}?", "Anything planned for {day}?", "What does my {day} look like?",
               "remind me what's on for {day}"]
NOTHING = ["nothing", "no event", "no plan", "free", "not have any", "don't have any", "no meeting", "empty",
           "no scheduled", "not scheduled"]


def gen_events(r, split):
    day = r.choice(list(EVENTS))
    c = call("get_events", day=day)
    shown = day.capitalize() if r.random() < 0.8 else day
    q = r.choice(EVENTS_TRAIN if split == "train" else EVENTS_TEST).format(day=shown)
    evs = EVENTS[day]
    if evs:
        ans = f"On {day.capitalize()} you have " + ", ".join(f"{title} at {tm}" for tm, title, _ in evs) + "."
        check = {"keywords": [k for _, _, k in evs]}
    else:
        ans = f"You have nothing on your calendar for {day.capitalize()}."
        check = {"keywords": [NOTHING]}
    return q, [[c]], ans, check


# ---------------- two-step and parallel ----------------
WC_TRAIN = [("What's the temperature in {city} in Fahrenheit?", "fahrenheit"),
            ("How warm is it in {city} in °F?", "fahrenheit"),
            ("Tell me the weather in {city} but in Fahrenheit.", "fahrenheit")]
WC_TEST = [("give me {city}'s temperature in fahrenheit", "fahrenheit"),
           ("{city} temp in F please", "fahrenheit"),
           ("What's the current temperature in {city} in kelvin?", "kelvin")]


def gen_weather_convert(r, split):
    city = pick_city(r, split)
    tmpl, unit = r.choice(WC_TRAIN if split == "train" else WC_TEST)
    c1 = call("get_weather", city=city_name(city))
    t = c1["result"]["temperature_c"]
    c2 = call("convert_units", value=t, from_unit="celsius", to_unit=unit)
    res = c2["result"]["result"]
    sym = "°F" if unit == "fahrenheit" else " K"
    ans = f"It's {t}°C in {city_name(city)} right now, which is {res}{sym}."
    return tmpl.format(city=show_city(r, city, split)), [[c1], [c2]], ans, {"numbers": [res]}


SC_TRAIN = ["How much would {n} shares of {t} cost?", "What's the cost of buying {n} {t} shares?"]
SC_TEST = ["I want to buy {n} shares of {name}. What's the total?", "price for {n} x {t} stock?"]


def gen_stock_calc(r, split):
    t = pick_ticker(r, split)
    n = r.randint(2, 250)
    c1 = call("get_stock_price", symbol=t)
    p = c1["result"]["price_usd"]
    c2 = call("calculator", expression=f"{n} * {p}")
    total = c2["result"]["result"]
    q = r.choice(SC_TRAIN if split == "train" else SC_TEST).format(n=n, t=t, name=c1["result"]["company"])
    ans = f"{n} shares of {t} at ${money(p)} would cost ${money(total)}."
    return q, [[c1], [c2]], ans, {"numbers": [total]}


PW_TRAIN = ["Compare the weather in {a} and {b}.", "What's the weather in {a} and {b}?"]
PW_TEST = ["how are conditions in {a} vs {b} right now?", "Check the weather for both {a} and {b}."]
PS_TRAIN = ["What are {a} and {b} trading at?", "Get me the prices of {a} and {b}."]
PS_TEST = ["{a} and {b} share prices?", "Look up both {a} and {b} for me."]


def gen_parallel(r, split):
    if r.random() < 0.5:
        a, b = pick_city(r, split), pick_city(r, split)
        while b == a:
            b = pick_city(r, split)
        c1, c2 = call("get_weather", city=city_name(a)), call("get_weather", city=city_name(b))
        w1, w2 = c1["result"], c2["result"]
        q = r.choice(PW_TRAIN if split == "train" else PW_TEST).format(a=show_city(r, a, split), b=show_city(r, b, split))
        ans = (f"{w1['city']} is {w1['temperature_c']}°C and {w1['condition']}, while "
               f"{w2['city']} is {w2['temperature_c']}°C and {w2['condition']}.")
        nums = [w1["temperature_c"], w2["temperature_c"]]
    else:
        a, b = pick_ticker(r, split), pick_ticker(r, split)
        while b == a:
            b = pick_ticker(r, split)
        c1, c2 = call("get_stock_price", symbol=a), call("get_stock_price", symbol=b)
        p1, p2 = c1["result"]["price_usd"], c2["result"]["price_usd"]
        q = r.choice(PS_TRAIN if split == "train" else PS_TEST).format(a=a, b=b)
        ans = f"{a} is trading at ${money(p1)} and {b} is at ${money(p2)}."
        nums = [p1, p2]
    return q, [[c1, c2]], ans, {"numbers": nums}


# ---------------- no tool needed ----------------
NO_TOOL_TRAIN = [
    ("Hi there!", "Hello! How can I help you today?"),
    ("Thanks, that's all.", "You're welcome! Have a great day."),
    ("How are you?", "I'm doing well, thanks for asking! What can I do for you?"),
    ("Good night!", "Good night! Sleep well."),
    ("What can you help me with?", "I can do calculations, convert units, check the weather, look up stock prices "
                                   "and read your calendar. I can also answer general questions."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("Why is the sky blue?", "Sunlight scatters off air molecules and blue light scatters the most "
                             "(Rayleigh scattering), so the sky looks blue."),
    ("What causes rain?", "Water vapour cools and condenses into droplets in clouds. When the droplets get heavy "
                          "enough they fall as rain."),
    ("What is a stock dividend?", "A dividend is a part of a company's profit that is paid out to its shareholders."),
    ("Explain what a kilometer is.", "A kilometer is a unit of length equal to 1000 meters."),
    ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Write a haiku about the ocean.", "Waves fold into foam\nthe moon pulls the tide away\nsalt wind on my face"),
    ("What does CPU stand for?", "CPU stands for Central Processing Unit."),
    ("What is photosynthesis?", "Photosynthesis is how plants use sunlight, water and carbon dioxide to make "
                                "glucose and release oxygen."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What language is spoken in Brazil?", "The official language of Brazil is Portuguese."),
    ("Give me a tip for staying focused while studying.", "Work in short blocks, like 25 minutes, with your phone "
                                                         "out of reach, then take a 5 minute break."),
    ("What is machine learning?", "Machine learning is a branch of AI where models learn patterns from data "
                                  "instead of being explicitly programmed."),
    ("How does a thermometer work?", "Most thermometers measure how a material, like liquid or a metal sensor, "
                                     "changes with temperature and map that change to a scale."),
    ("What is the difference between weather and climate?", "Weather is the short-term state of the atmosphere, "
                                                            "while climate is the average weather over many years."),
    ("What is a bear market?", "A bear market is when stock prices fall 20% or more from recent highs, usually "
                               "with a lot of pessimism."),
    ("What is the boiling point of water at sea level?", "Water boils at 100°C (212°F) at sea level."),
    ("Why do we have seasons?", "Because the Earth's axis is tilted, so each hemisphere gets more direct sunlight "
                                "at different times of the year."),
    ("Suggest a name for a pet cat.", "How about Miso? Other options are Luna, Pepper or Mochi."),
    ("Translate 'good morning' into Spanish.", "'Good morning' in Spanish is 'Buenos días'."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How do I make a cup of tea?", "Boil water, pour it over a tea bag, let it steep for 3 to 5 minutes, then "
                                    "add milk or sugar if you like."),
    ("What is an API?", "An API (Application Programming Interface) is a set of rules that lets one program talk "
                        "to another."),
    ("Who was the first person to walk on the moon?", "Neil Armstrong was the first person to walk on the moon, "
                                                      "in 1969."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What is a calendar used for?", "A calendar is used to organise days, weeks and months and to keep track of "
                                     "events and appointments."),
    ("What day comes after Wednesday?", "Thursday comes after Wednesday."),
    ("Recommend a good book.", "Try 'Sapiens' by Yuval Noah Harari if you like non-fiction, or 'The Hobbit' for "
                               "a fun story."),
    ("What is the speed of light?", "The speed of light is about 299,792 km per second."),
    ("Define humidity.", "Humidity is the amount of water vapour in the air."),
    ("What is inflation?", "Inflation is the general rise in prices over time, which lowers what money can buy."),
    ("How many continents are there?", "There are seven continents."),
    ("What's the opposite of 'ancient'?", "The opposite of 'ancient' is 'modern'."),
    ("Tell me a fun fact about octopuses.", "Octopuses have three hearts and blue blood."),
    ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32°F."),
    ("Is a tomato a fruit or a vegetable?", "Botanically a tomato is a fruit, but it's usually cooked as a vegetable."),
    ("What is Python used for?", "Python is used for web development, data analysis, machine learning, "
                                 "scripting and automation."),
    ("What does 'meteorology' mean?", "Meteorology is the science that studies the atmosphere and weather."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone."),
    ("How should I prepare for an exam?", "Start early, revise with past papers, test yourself instead of "
                                          "rereading, and get enough sleep before the exam."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
    ("Name three primary colors.", "Red, blue and yellow."),
    ("What is the currency of Japan?", "The currency of Japan is the yen."),
    ("Why do leaves change colour in autumn?", "Trees stop making chlorophyll, so the green fades and the yellow "
                                               "and red pigments show."),
    ("What's the difference between a stock and a bond?", "A stock is a share of ownership in a company, while a "
                                                          "bond is a loan you give to a company or government."),
    ("Write a one-line motivational quote.", "Small steps every day add up to big results."),
    ("What is gravity?", "Gravity is the force that pulls objects with mass towards each other."),
    ("Which ocean is the largest?", "The Pacific Ocean is the largest."),
    ("What does a barometer measure?", "A barometer measures atmospheric pressure."),
    ("What is the main ingredient in guacamole?", "The main ingredient in guacamole is avocado."),
    ("Explain what a time zone is.", "A time zone is a region that uses the same standard time, usually based on "
                                     "its offset from UTC."),
    ("What is a leap year?", "A leap year has 366 days, with an extra day added to February."),
    ("What does a share of stock represent?", "A share represents a small piece of ownership in a company."),
    ("Who is known as the father of computers?", "Charles Babbage is known as the father of computers."),
]
NO_TOOL_PREFIX = ["", "", "", "Hey, ", "Quick question: ", "Hi! ", "Can you tell me this: "]

# test questions: (question, list of keyword alternatives that the answer should contain)
NO_TOOL_TEST = [
    ("hello! how's it going?", []),
    ("Who discovered penicillin?", [["fleming"]]),
    ("What is the capital city of Australia?", [["canberra"]]),
    ("why does it get colder at higher altitudes?", []),
    ("What does a stock exchange do?", []),
    ("In which country are the pyramids of Giza?", [["egypt"]]),
    ("Write a short poem about coffee.", []),
    ("What is the plural of 'mouse'?", [["mice"]]),
    ("What is the metric system?", []),
    ("How many legs does a spider have?", [["8", "eight"]]),
    ("Which days of the week are the weekend?", [["saturday"], ["sunday"]]),
    ("What is the chemical formula for water?", [["h2o", "h₂o"]]),
    ("Can you give me some advice on saving money?", []),
    ("Who wrote the Harry Potter books?", [["rowling"]]),
    ("What is a hurricane?", []),
    ("What planet is known as the Red Planet?", [["mars"]]),
    ("thanks for the help!", []),
    ("What does GPU stand for?", [["graphics"]]),
    ("Tell me something interesting about elephants", []),
    ("What is the longest river in the world?", [["nile", "amazon"]]),
]


def gen_none(r, split):
    q, a = r.choice(NO_TOOL_TRAIN)
    p = r.choice(NO_TOOL_PREFIX)
    if p:
        q = p + q[0].lower() + q[1:]
    return q, [], a, {}


GENERATORS = {"calc": gen_calc, "convert": gen_convert, "weather": gen_weather, "stock": gen_stock,
              "events": gen_events, "weather_convert": gen_weather_convert, "stock_calc": gen_stock_calc,
              "parallel": gen_parallel, "none": gen_none}


def build(counts, split, r, prefix, seen):
    rows = []
    for kind, n in counts.items():
        if kind == "none" and split == "test":
            for q, kw in NO_TOOL_TEST[:n]:
                rows.append({"kind": kind, "query": q, "steps": [], "answer": None, "check": {"keywords": kw}})
            continue
        made, tries = 0, 0
        while made < n:
            q, steps, ans, check = GENERATORS[kind](r, split)
            tries += 1
            if q in seen and tries < 50 * n:
                continue
            seen.add(q)
            rows.append({"kind": kind, "query": q, "steps": steps, "answer": ans, "check": check})
            made += 1
    r.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"{prefix}-{i:04d}"
    return rows


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    r = random.Random(SEED)
    seen = set()
    train = build(N_TRAIN, "train", r, "train", seen)
    val_counts = {k: max(1, round(v * N_VAL / sum(N_TRAIN.values()))) for k, v in N_TRAIN.items()}
    val = build(val_counts, "train", r, "val", seen)
    test = build(N_TEST, "test", r, "test", set())
    overlap = {x["query"] for x in test} & {x["query"] for x in train + val}
    assert not overlap, overlap
    write(train, "data/train.jsonl")
    write(val, "data/val.jsonl")
    write(test, "data/test.jsonl")
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        kinds = {}
        for x in rows:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        print(name, len(rows), kinds)
