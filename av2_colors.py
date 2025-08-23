# A color look up table (LUT) for various classes of objects, designed for bright,
# vibrant, and unique identification with semantic associations.
# The colors are represented in RGB tuples (0-255).

AV2_COLORS = {
    # Transportation
    "REGULAR_VEHICLE": (
        60,
        180,
        75,
    ),  # 💚 A vibrant green, similar to "go" on a traffic light.
    "LARGE_VEHICLE": (0, 130, 200),  # 💙 A bold blue to signify large, stable objects.
    "BICYCLE": (
        255,
        255,
        25,
    ),  # 💛 A bright yellow, highly visible like a cyclist's jersey.
    "BUS": (255, 150, 25),  # 🧡 A bright orange, a common color for public transport.
    "SCHOOL_BUS": (255, 200, 0),  # 💛 A classic school bus yellow, unmistakable.
    "BOX_TRUCK": (100, 140, 255),  # 💙 A light, prominent blue.
    "TRUCK": (
        128,
        0,
        128,
    ),  # 💜 A distinct purple, differentiating it from other vehicles.
    "MOTORCYCLE": (
        255,
        10,
        245,
    ),  # 💖 A vibrant magenta to represent speed and agility.
    "VEHICULAR_TRAILER": (
        190,
        190,
        190,
    ),  # 🩶 A light gray, representing a secondary, towed object.
    "TRUCK_CAB": (
        100,
        0,
        100,
    ),  # 💜 A deeper purple to signify the main part of a semi-truck.
    "ARTICULATED_BUS": (
        200,
        10,
        20,
    ),  # ❤️ A dark red, differentiating it from a regular bus.
    "RAILED_VEHICLE": (165, 42, 42),  # 🤎 A warm brown, like the rust of an old train.
    "WHEELED_DEVICE": (
        150,
        200,
        255,
    ),  # 🩵 A light, sky-blue for personal, non-conventional devices.
    # People and Animals
    "PEDESTRIAN": (
        255,
        60,
        60,
    ),  # ❤️ A bold red, universally associated with "stop" or caution for people.
    "BICYCLIST": (
        255,
        165,
        0,
    ),  # 🧡 A bright orange, combining the boldness of a cyclist's vest with the red for people.
    "MOTORCYCLIST": (
        255,
        125,
        255,
    ),  # 🩷 A soft magenta, a lighter version of the motorcycle color.
    "WHEELED_RIDER": (
        0,
        255,
        255,
    ),  # 🩵 A bright cyan, for people on non-traditional wheels.
    "DOG": (
        255,
        140,
        0,
    ),  # 🟠 A deep orange, a common color for a dog's collar or vest.
    "ANIMAL": (139, 69, 19),  # 🤎 A brown color to represent natural wildlife.
    "OFFICIAL_SIGNALER": (
        0,
        255,
        0,
    ),  # 🟢 A bright green, like a "go" sign, to signify control and authority.
    "WHEELCHAIR": (
        112,
        128,
        144,
    ),  # 🩶 A slate gray, similar to the color of many wheelchairs.
    "STROLLER": (
        255,
        192,
        203,
    ),  # 🩷 A gentle pink, commonly associated with babies and children.
    # Road and Construction Objects
    "BOLLARD": (153, 51, 255),  # 💜 A vibrant purple for a fixed, geometric object.
    "CONSTRUCTION_CONE": (255, 69, 0),  # 🧡 A classic safety orange.
    "CONSTRUCTION_BARREL": (
        255,
        140,
        0,
    ),  # 🟠 A slightly different shade of orange from the cone.
    "STOP_SIGN": (255, 0, 0),  # 🟥 The iconic red of a stop sign.
    "SIGN": (255, 215, 0),  # 🟡 A vibrant gold, representing important information.
    "MESSAGE_BOARD_TRAILER": (
        100,
        100,
        100,
    ),  # 🩶 A neutral gray for an electronic device.
    "MOBILE_PEDESTRIAN_SIGN": (
        135,
        206,
        250,
    ),  # 🩵 A light blue, representing guidance for people.
    "TRAFFIC_LIGHT_TRAILER": (
        128,
        128,
        0,
    ),  # 🟤 A mix of yellow and green, for a temporary light.
}
