"""Built-in reading passages: natural prose, roughly 45-75 words, no digits
(numbers make the text-to-phone mapping ambiguous). Each mixes many sounds;
together they cover th, v/w, r/l, z, sh/ch/j, vowel pairs and word endings."""

BANK = [
    "I think the weather will be very warm this weekend, so we should visit the village by the river. "
    "My brother thought it would rain, but the forecast says there is only a small chance. "
    "If it does, we can always wait in the little café near the bridge and watch the boats go by.",

    "The library on the corner closes early on Thursdays, which is annoying when you need a quiet place "
    "to work. Last week I arrived at half past four and found the doors already locked. "
    "A woman with a stack of books shrugged and told me the schedule had changed without any warning.",

    "Every morning my neighbor walks his two dogs along the harbor, rain or shine. "
    "One of them is old and slow, and the other pulls so hard on the leash that he almost falls over. "
    "He says the walk clears his head better than coffee, though I notice he usually carries a cup anyway.",

    "Learning to cook was harder than I expected. The first time I made bread, it came out flat and heavy, "
    "like a brick. My grandmother laughed and showed me how to knead the dough properly, folding it over "
    "and over until it felt smooth and alive. These days the kitchen smells wonderful every Sunday.",

    "We drove north for three hours through thick fog before the sky finally cleared. "
    "The mountains appeared all at once, enormous and silent, with snow still clinging to the highest ridges. "
    "Nobody in the car said a word for a while. Then my sister whispered that it looked like a painting.",

    "Our team spent most of the year rebuilding the old system from scratch. It was slow, careful work, "
    "and there were weeks when nothing seemed to move forward. But when we finally switched over, "
    "everything ran faster and the error reports almost disappeared. The relief in the office was obvious.",

    "She bought a used violin at a market in the old town and started taking lessons twice a week. "
    "At first the sound was awful, more like a cat than music. Her roommates were patient, mostly. "
    "After six months she could play a simple waltz, and now they ask her to perform at every party.",

    "The museum has a whole room devoted to clocks. Some are tiny enough to wear on a ring, "
    "while others are taller than a person and chime loudly every quarter hour. "
    "Standing among them, you hear a constant soft ticking, as though the walls themselves were breathing.",

    "Thank you all for coming tonight. I know it was difficult for many of you to travel this far, "
    "especially with the trains delayed. We wanted to gather the people who made this project possible, "
    "and to celebrate what we achieved together. Please enjoy the food, and don't leave without dessert.",

    "When the power went out, the whole street went quiet. People came out of their houses with candles "
    "and flashlights, and for once everybody started talking to each other. Someone brought out a guitar. "
    "By the time the lights came back on, nobody really wanted to go back inside.",

    "The fisherman checked his nets with rough, careful hands. The catch was smaller than last season, "
    "and he worried about the price at the market. Still, he whistled as he worked, because the water "
    "was calm and the light on the waves was the most beautiful thing he knew.",

    "My first apartment was so small that I could cook dinner while lying in bed. The heating worked "
    "only when it felt like it, and the window above the sink would not close properly in winter. "
    "Even so, I remember it fondly. It was the first place that was truly mine.",

    "Scientists have discovered that some birds can remember thousands of hiding places for their food. "
    "They bury seeds in the autumn and find them again months later, even under deep snow. "
    "Researchers believe the birds use landmarks, like trees and rocks, to build a kind of mental map.",

    "It's usually the small, ordinary things that make a journey memorable. A conversation with a stranger "
    "on a bus, a meal in a crowded kitchen, a wrong turn that leads somewhere unexpected. "
    "The famous sights are lovely, of course, but they rarely surprise you the way people do.",

    "The children were supposed to be asleep, but we could hear them whispering and giggling upstairs. "
    "Eventually their father went up and told them a long, boring story about tax forms. "
    "Within ten minutes the house was silent. He came back down looking very pleased with himself.",

    "Our village has a strange tradition every winter. On the shortest day of the year, everyone gathers "
    "by the frozen lake and lights a huge fire. We sing old songs whose words nobody fully understands, "
    "and then we walk home together in the dark, which somehow never feels cold.",

    "A good presentation is less about the slides and more about the story. Start with the question "
    "you want to answer, show the evidence clearly, and finish with what you think should happen next. "
    "Speak slowly, look at the audience, and leave enough time for their questions at the end.",

    "The bakery on our street opens at six, and by seven there is already a line outside. "
    "The owner works through the night, shaping loaves and filling pastries with chocolate and cherries. "
    "She says she has not slept past five in twenty years, and that she would not change a thing.",

    "We were halfway up the hill when the thunder started. Within seconds the rain was so heavy we could "
    "barely see the path. We ran for a shelter, soaked through and laughing, and waited there with a family "
    "of tourists who shared their sandwiches and told us about their travels.",

    "Some people relax by reading, others by running. My father relaxes by fixing things that are not "
    "broken. He will spend a whole weekend taking apart a perfectly good radio, cleaning every piece, "
    "and putting it back together. Usually there are a few screws left over, which worries my mother.",

    "The village well was the center of everything long before anyone had running water. Women would "
    "walk there with heavy buckets, trading news and gossip while they waited. When the pipes finally "
    "arrived, many of them said they missed those mornings more than they expected.",

    "Why do we find it so hard to throw old things away? A drawer full of cables, a jacket that no longer "
    "fits, a box of letters from friends we have not seen in years. Perhaps we are not keeping the objects "
    "at all, but the versions of ourselves that used them.",

    "The train was nearly empty, so I chose a seat by the window and watched the fields slide past. "
    "Cows, a church tower, a farmer on a red tractor, then a long stretch of forest. "
    "Somewhere along the way I fell asleep, and woke up just as we pulled into the station.",

    "Weather forecasts have become remarkably accurate over the last few decades. Satellites watch the "
    "clouds from above, while thousands of stations measure wind, pressure and temperature on the ground. "
    "Powerful computers combine all of this information to predict whether you will need an umbrella tomorrow.",
]

import os as _os
if _os.environ.get("PRONOUNCE_LANG", "en") == "nb":
    from paragraphs_nb import BANK  # noqa: F811
elif _os.environ.get("PRONOUNCE_LANG", "en") == "el":
    from paragraphs_el import BANK  # noqa: F811

# bank_<lang>.json: passages with their reference scores precomputed (tools/bank_build.py).
# English: real public-domain passages (tools/bank_candidates.py → bank_judge.py → bank_build.py).
# Norwegian, Greek: the hand-written passages (paragraphs_nb/el), precomputed for the hosted version (no TTS on the server).
# META maps text -> author, title, setup, areas, ref.
META = {}
if _os.environ.get("PRONOUNCE_LANG", "en") in ("en", "nb", "el"):
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       f"bank_{_os.environ.get('PRONOUNCE_LANG', 'en')}.json")
    if _os.path.exists(_p):
        import json as _json
        with open(_p, encoding="utf-8") as _f:
            META = {b["text"]: b for b in _json.load(_f)}
        if META:
            BANK = list(META)
