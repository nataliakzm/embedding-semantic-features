danger_pairs = [
    ("He is falling into an ice hole", "He is looking at an ice hole"),
    ("He is taking a broken elevator", "He is fixing a broken elevator"),
    ("He got a cramp while swimming", "He got a cramp while jogging"),
    ("He is hiking a mountain wearing flip-flops", "He is hiking a mountain wearing a fisherman hat"),
    ("She is touching a broken glass", "She is touching a broken doll"),
    ("She is using a hairdryer in the bathtub", "She is using a hairdryer in bed"),
    ("She is walking on the edge of a cliff", "She is walking on the side of a cliff"),
    ("He is texting while driving his car", "He is texting while attending the conference"),
    ("She is using her phone while taking a bath", "She is using her phone while taking a sunbath"),
    ("He is crossing the street without looking", "He is crossing the street without talking"),
    ("She is feeding bears in the forest", "She is feeding bears at the zoo"),
    ("He is showing a red flag to the bull", "He is showing a red flag to the elephant"),
    ("She is touching a hot stove", "She is touching a cold stove"),
    ("He is walking over thin ice", "He is walking over thick ice"),
    ("The girl is walking alone on a dark street", "The man is walking alone on a dark street"),
    ("He is using a chainsaw without safety gear", "He is using a chainsaw with safety gear"),
    ("He is driving without seatbelts", "He is driving with seatbelts"),
    ("She is sunbathing without sunscreen", "She is sunbathing with sunscreen"),
    ("He is driving at night without headlights", "He is driving at night with headlights"),
    ("He is riding a motorcycle without a helmet", "He is riding a motorcycle with a helmet"),
    ("He is walking on the railroad tracks", "He is walking at the railroad station"),
    ("He is playing with a loaded gun", "He is playing with an unloaded gun"),
    ("He is passing a red light", "He is waiting at a red light"),
    ("He is climbing a wet ladder", "He is climbing a dry ladder"),
    ("He is touching a live wire", "He is touching a disabled wire"),
    ("The pregnant woman is eating raw meat", "The pregnant woman is eating overcooked meat"),
    ("She is playing with animals in the forest", "She is playing with animals on the farm"),
    ("He is eating fruits without washing them", "He is washing fruits before eating them"),
    ("He is charging his phone while taking a bath", "He is charging his phone while sleeping"),
    ("The child is playing with small pieces of plastic", "The man is playing with small pieces of plastic"),
    ("He is driving fast in a small village", "He is driving fast on the freeway"),
    ("The child is taking candy from a stranger", "The child is taking candy from a relative"),
    ("She is cutting her finger", "She is cutting her shirt"),
    ("The boy is fixing live wires", "The electrician is fixing live wires"),
    ("He is standing over a railing", "He is standing behind a railing"),
    ("He is jumping from the bridge", "He is jumping on the elastic mat"),
    ("She didn't see the bus coming", "She didn't see the boy coming"),
    ("He is lighting a fire in a dry field", "He is lighting a fire in a wet field"),
    ("He is running downhill with untied shoelaces", "He is running downhill with dirty shoelaces"),
    ("He runs on the street", "He runs in the park"),
    ("She is smoking at a petrol station", "She is smoking at the steel mill"),
    ("The child is near the pool", "The child is near the fountain"),
    ("He is driving with bald tires on a frozen road", "He is driving with bald tires on a hot road"),
    ("She is fixing an unplugged TV", "She is fixing a plugged TV"),
    ("He is hiking in the mountains without a map", "He is hiking in the mountains with a map"),
    ("She got lost in the mountains in the evening", "She got lost in the mountains in the morning"),
    ("He is driving in the fog without lights", "He is driving in the fog with lights"),
    ("She is pulling a tray out of the stove without a mitt", "She is putting a tray in the stove with a mitt"),
    ("He is driving a car after drinking alcohol", "He is taking a cab after drinking alcohol"),
    ("She is burning candles before leaving the room", "She is blowing out candles before leaving the room"),
    ("He is leaving the stove on before heading out", "He is leaving the stove off before heading out"),
    ("She is walking near moving cars", "She is walking near parked cars"),
    ("He is ignoring a storm warning before camping", "He is avoiding camping due to a storm warning"),
    ("He is heading to the beach after a tsunami alert", "He is heading to the hill after a tsunami alert"),
    ("She is drinking dirty water", "She is pouring dirty water"),
    ("He is swimming against a rip current", "He is swimming perpendicularly to a rip current"),
    ("She falls into the sea from the sailing boat", "She falls into the sea from the surfboard"),
    ("She is skiing off track", "She is skiing out of town"),
    ("She is bicycling without hands", "She is bicycling without gloves"),
    ("He is lighting a fire in the woods", "He is lighting a fire in the fireplace"),
    ("She is taking selfies on a skyscraper ledge", "She is taking selfies with a skyscraper in the background"),
    ("The young boy is cutting meat with a knife", "The old man is cutting meat with a knife"),
    ("He fell asleep in the dinghy", "She fell asleep on the bench"),
    ("He is cleaning a live socket with a wet cloth", "He is cleaning a live socket with a dry cloth"),
    ("He drives while looking at his girlfriend", "He drives while looking at the road"),
    ("She is wearing wet clothes at the North Pole", "She is wearing dry clothes at the North Pole"),
    ("He crossed the intersection at a red light", "He crossed the intersection at a green light"),
    ("He leans against train doors", "He stands beside train doors"),
    ("He sleeps near the river during intense rainfall", "He sleeps near the river during fireworks"),
    ("She holds the scissors close to her eye", "She holds the scissors close to her foot"),
    ("She ignores the railing while climbing", "She holds the railing while climbing"),
    ("She crosses roads without checking", "She crosses roads after checking"),
    ("She hurries across icy sidewalks", "She hurries across hot sidewalks"),
    ("She sleeps with contact lenses on", "She sleeps with contact lenses off"),
    ("She walks on the wet floor", "She walks on the dry floor"),
    ("He touches broken glasses", "He touches broken spoons"),
    ("He drives along the highway sleepily", "He drives along the highway worried"),
    ("He eats undercooked meat", "He eats undercooked pasta"),
    ("He drives fast along the winding road", "He drives fast along the straight road"),
    ("She crosses the road outside the crosswalk", "She crosses the road on the crosswalk"),
    ("The dog crosses the road", "The man crosses the road"),
    ("He is hanging from a thin branch", "He is hanging from a thick branch"),
    ("He downloaded an unknown app", "He downloaded an expensive app"),
    ("He lost his house key", "He lost his house broom"),
    ("He left the stove unattended", "He left the dishwasher unattended"),
    ("She leaned too far over the balcony", "She leaned too far over the pool"),
    ("They stayed in the sun too long", "They stayed in the sun too little"),
    ("They hiked off the marked trail", "They hiked on the marked trail"),
    ("They left their drinks unattended at the bar", "They left their coats unattended at the bar"),
    ("She didn't lock the door at night", "She didn’t lock the fridge at night"),
    ("She put profits before safety", "She put safety before profits"),
    ("They ate raw shellfish at the roadside stand", "They ate raw shellfish at the restaurant"),
    ("He dived into the pool without checking the depth", "He dived into the pool after checking the depth"),
    ("He postponed the physical exam", "He postponed the holiday"),
    ("He is trying to take a book from the high shelf", "He is trying to take a book from the bottom shelf"),
    ("He left the faucet open", "He left the bathroom open"),
    ("She doesn't drink after exercising", "She doesn't drink after dinner"),
    ("She is standing on a wheelchair", "She sits in a wheelchair"),
    ("He smokes in bed", "He smokes on the terrace"),
    ("She is crossing the street looking one way", "She is crossing the street looking both ways"),
]

danger_category = "He is falling into an ice hole, He is looking at an ice hole, He is taking a broken elevator, He is fixing a broken elevator, He got a cramp while swimming, He got a cramp while jogging, He is hiking a mountain wearing flip-flops, He is hiking a mountain wearing a fisherman hat, She is touching a broken glass, She is touching a broken doll, She is using a hairdryer in the bathtub, She is using a hairdryer in bed, She is walking on the edge of a cliff, She is walking on the side of a cliff, He is texting while driving his car, He is texting while attending the conference, She is using her phone while taking a bath, She is using her phone while taking a sunbath, He is crossing the street without looking, He is crossing the street without talking, She is feeding bears in the forest, She is feeding bears at the zoo, He is showing a red flag to the bull, He is showing a red flag to the elephant, She is touching a hot stove, She is touching a cold stove, He is walking over thin ice, He is walking over thick ice, The girl is walking alone on a dark street, The man is walking alone on a dark street, He is using a chainsaw without safety gear, He is using a chainsaw with safety gear, He is driving without seatbelts, He is driving with seatbelts, She is sunbathing without sunscreen, She is sunbathing with sunscreen, He is driving at night without headlights, He is driving at night with headlights, He is riding a motorcycle without a helmet, He is riding a motorcycle with a helmet, He is walking on the railroad tracks, He is walking at the railroad station, He is playing with a loaded gun, He is playing with an unloaded gun, He is passing a red light, He is waiting at a red light, He is climbing a wet ladder, He is climbing a dry ladder, He is touching a live wire, He is touching a disabled wire, The pregnant woman is eating raw meat, The pregnant woman is eating overcooked meat, She is playing with animals in the forest, She is playing with animals on the farm, He is eating fruits without washing them, He is washing fruits before eating them, He is charging his phone while taking a bath, He is charging his phone while sleeping, The child is playing with small pieces of plastic, The man is playing with small pieces of plastic, He is driving fast in a small village, He is driving fast on the freeway, The child is taking candy from a stranger, The child is taking candy from a relative, She is cutting her finger, She is cutting her shirt, The boy is fixing live wires, The electrician is fixing live wires, He is standing over a railing, He is standing behind a railing, He is jumping from the bridge, He is jumping on the elastic mat, She didn't see the bus coming, She didn't see the boy coming, He is lighting a fire in a dry field, He is lighting a fire in a wet field, He is running downhill with untied shoelaces, He is running downhill with dirty shoelaces, He runs on the street, He runs in the park, She is smoking at a petrol station, She is smoking at the steel mill, The child is near the pool, The child is near the fountain, He is driving with bald tires on a frozen road, He is driving with bald tires on a hot road, She is fixing an unplugged TV, She is fixing a plugged TV, He is hiking in the mountains without a map, He is hiking in the mountains with a map, She got lost in the mountains in the evening, She got lost in the mountains in the morning, He is driving in the fog without lights, He is driving in the fog with lights, She is pulling a tray out of the stove without a mitt, She is putting a tray in the stove with a mitt, He is driving a car after drinking alcohol, He is taking a cab after drinking alcohol, She is burning candles before leaving the room, She is blowing out candles before leaving the room, He is leaving the stove on before heading out, He is leaving the stove off before heading out, She is walking near moving cars, She is walking near parked cars, He is ignoring a storm warning before camping, He is avoiding camping due to a storm warning, He is heading to the beach after a tsunami alert, He is heading to the hill after a tsunami alert, She is drinking dirty water, She is pouring dirty water, He is swimming against a rip current, He is swimming perpendicularly to a rip current, She falls into the sea from the sailing boat, She falls into the sea from the surfboard, She is skiing off track, She is skiing out of town, She is bicycling without hands, She is bicycling without gloves, He is lighting a fire in the woods, He is lighting a fire in the fireplace, She is taking selfies on a skyscraper ledge, She is taking selfies with a skyscraper in the background, The young boy is cutting meat with a knife, The old man is cutting meat with a knife, He fell asleep in the dinghy, She fell asleep on the bench, He is cleaning a live socket with a wet cloth, He is cleaning a live socket with a dry cloth, He drives while looking at his girlfriend, He drives while looking at the road, She is wearing wet clothes at the North Pole, She is wearing dry clothes at the North Pole, He crossed the intersection at a red light, He crossed the intersection at a green light, He leans against train doors, He stands beside train doors, He sleeps near the river during intense rainfall, He sleeps near the river during fireworks, She holds the scissors close to her eye, She holds the scissors close to her foot, She ignores the railing while climbing, She holds the railing while climbing, She crosses roads without checking, She crosses roads after checking, She hurries across icy sidewalks, She hurries across hot sidewalks, She sleeps with contact lenses on, She sleeps with contact lenses off, She walks on the wet floor, She walks on the dry floor, He touches broken glasses, He touches broken spoons, He drives along the highway sleepily, He drives along the highway worried, He eats undercooked meat, He eats undercooked pasta, He drives fast along the winding road, He drives fast along the straight road, She crosses the road outside the crosswalk, She crosses the road on the crosswalk, The dog crosses the road, The man crosses the road, He is hanging from a thin branch, He is hanging from a thick branch, He downloaded an unknown app, He downloaded an expensive app, He lost his house key, He lost his house broom, He left the stove unattended, He left the dishwasher unattended, She leaned too far over the balcony, She leaned too far over the pool, They stayed in the sun too long, They stayed in the sun too little, They hiked off the marked trail, They hiked on the marked trail, They left their drinks unattended at the bar, They left their coats unattended at the bar, She didn't lock the door at night, She didn’t lock the fridge at night, She put profits before safety, She put safety before profits, They ate raw shellfish at the roadside stand, They ate raw shellfish at the restaurant, He dived into the pool without checking the depth, He dived into the pool after checking the depth, He postponed the physical exam, He postponed the holiday, He is trying to take a book from the high shelf, He is trying to take a book from the bottom shelf, He left the faucet open, He left the bathroom open, She doesn't drink after exercising, She doesn't drink after dinner, She is standing on a wheelchair, She sits in a wheelchair, He smokes in bed, He smokes on the terrace, She is crossing the street looking one way, She is crossing the street looking both ways"

size_pairs = [
("a conference table", "a coffee table"),
("a dining table", "a tea table"),
("a banquet table", "a coffee table"),
("an executive desk", "a study desk"),
("a soccer ball", "a billiard ball"),
("a soccer ball", "a ping pong ball"),
("a basket ball", "a golf ball"),
("a volleyball", "a tennis ball"),
("a bowling ball", "a billiard ball"),
("a soccer ball", "a cricket ball"),
("a sleeping bag", "a tea bag"),
("a cement bag", "a paper bag"),
("a golf bag", "a lunch bag"),
("a laundry bag", "a coin bag"),
("a laundry bag", "a tea bag"),
("a checked bag", "a carry-on bag"),
("a grocery bag", "a lunch bag"),
("a serving dish", "a soap dish"),
("a soup spoon", "a tea spoon"),
("a serving spoon", "a sugar spoon"),
("a bread knife", "a butter knife"),
("a beer glass", "a whiskey glass"),
("a moving box", "a lunch box"),
("a freight train", "a toy train"),
("a passenger train", "a model train"),
("a house door", "a garage door"),
("a garage door", "a cabinet door"),
("a stage lamp", "a task lamp"),
("a floor lamp", "a book lamp"),
("a floor lamp", "a desk lamp"),
("a street lamp", "a bedside lamp"),
("a beach towel", "a paper towel"),
("a pool towel", "a hand towel"),
("a tower clock", "a wall clock"),
("a water bottle", "a pill bottle"),
("a knitting needle", "a sewing needle"),
("a knitting needle", "a tattoo needle"),
("a shopping purse", "a change purse"),
("gardening scissors", "tailor's scissors"),
("garden scissors", "nail scissors"),
("a pneumatic drill", "a hand drill"),
("a soup bowl", "a dip bowl"),
("a salad bowl", "a soy bowl"),
("a flower pot", "a tea pot"),
("a garden pot", "a seed pot"),
("a cooking pot", "a tea pot"),
("a yoga mat", "a bath mat"),
("a floor mat", "a coaster mat"),
("a firefighter ladder", "a step ladder"),
("a library ladder", "a bed ladder"),
("a beach umbrella", "a cocktail umbrella"),
("a patio umbrella", "a normal umbrella"),
("a magazine rack", "a spice rack"),
("ski glasses", "reading glasses"),
("a church bell", "a school bell"),
("a cargo plane", "a model plane"),
("an electric guitar", "a toy guitar"),
("a cookie tin", "a mint tin"),
("a pasta strainer", "a tea strainer"),
("a beer mug", "a tea mug"),
("a sofa cushion", "a pin cushion"),
("a carpet roll", "a paper roll"),
("a waste container", "a tea container"),
("a crane hook", "a wall hook"),
("a shopping cart", "a tea cart"),
("a beach house", "a doll house"),
("a gaming monitor", "a portable monitor"),
("a bench grinder", "a hand grinder"),
("a dining set", "a tea set"),
("a garden fork", "a dinner fork"),
("a ceiling fan", "a desk fan"),
("an industrial scale", "a kitchen scale"),
("a concert speaker", "a home speaker"),
("an event tent", "a camping tent"),
("an industrial fridge", "a home fridge"),
("a wall map", "a pocket map"),
("a wall calendar", "a desk calendar"),
("a stage microphone", "a cellular microphone"),
("an electric broom", "a hand broom"),
("a patio heater", "a personal heater"),
("a tool case", "a pencil case"),
("a horse saddle", "a bicycle saddle"),
("a fishing net", "a butterfly net"),
("an olympic pool", "an inflatable pool"),
("a swimming pool", "a fish pool"),
("a bottle of wine", "a bottle of perfume"),
("a ping pong table", "a bedside table"),
("an office table", "a coffee table"),
("a gym bag", "an evening bag"),
("a wardrobe box", "a jewelry box"),
("a winde glass", "a cordiale glass"),
("a serving dish", "a dessert dish"),
("a milk cup", "an espresso cup"),
("a floor vase", "a bud vase"),
("a surf board", "a cutting board"),
("a walking stick", "a cotton stick"),
("a hot air balloon", "a party balloon"),
("a military tent", "a camping tent"),
("a traffic cone", "an ice scream cone"),
("a laundry bin", "a desk bin"),
]

size_category = "a conference table, a coffee table, a dining table, a tea table, a banquet table, a coffee table, an executive desk, a study desk, a soccer ball, a billiard ball, a soccer ball, a ping pong ball, a basket ball, a golf ball, a volleyball, a tennis ball, a bowling ball, a billiard ball, a soccer ball, a cricket ball, a sleeping bag, a tea bag, a cement bag, a paper bag, a golf bag, a lunch bag, a laundry bag, a coin bag, a laundry bag, a tea bag, a checked bag, a carry-on bag, a grocery bag, a lunch bag, a serving dish, a soap dish, a soup spoon, a tea spoon, a serving spoon, a sugar spoon, a bread knife, a butter knife, a beer glass, a whiskey glass, a moving box, a lunch box, a freight train, a toy train, a passenger train, a model train, a house door, a garage door, a garage door, a cabinet door, a stage lamp, a task lamp, a floor lamp, a book lamp, a floor lamp, a desk lamp, a street lamp, a bedside lamp, a beach towel, a paper towel, a pool towel, a hand towel, a tower clock, a wall clock, a water bottle, a pill bottle, a knitting needle, a sewing needle, a knitting needle, a tattoo needle, a shopping purse, a change purse, gardening scissors, tailor's scissors, garden scissors, nail scissors, a pneumatic drill, a hand drill, a soup bowl, a dip bowl, a salad bowl, a soy bowl, a flower pot, a tea pot, a garden pot, a seed pot, a cooking pot, a tea pot, a yoga mat, a bath mat, a floor mat, a coaster mat, a firefighter ladder, a step ladder, a library ladder, a bed ladder, a beach umbrella, a cocktail umbrella, a patio umbrella, a normal umbrella, a magazine rack, a spice rack, ski glasses, reading glasses, a church bell, a school bell, a cargo plane, a model plane, an electric guitar, a toy guitar, a cookie tin, a mint tin, a pasta strainer, a tea strainer, a beer mug, a tea mug, a sofa cushion, a pin cushion, a carpet roll, a paper roll, a waste container, a tea container, a crane hook, a wall hook, a shopping cart, a tea cart, a beach house, a doll house, a gaming monitor, a portable monitor, a bench grinder, a hand grinder, a dining set, a tea set, a garden fork, a dinner fork, a ceiling fan, a desk fan, an industrial scale, a kitchen scale, a concert speaker, a home speaker, an event tent, a camping tent, an industrial fridge, a home fridge, a wall map, a pocket map, a wall calendar, a desk calendar, a stage microphone, a cellular microphone, an electric broom, a hand broom, a patio heater, a personal heater, a tool case, a pencil case, a horse saddle, a bicycle saddle, a fishing net, a butterfly net, an olympic pool, an inflatable pool, a swimming pool, a fish pool, a bottle of wine, a bottle of perfume, a ping pong table, a bedside table, an office table, a coffee table, a gym bag, an evening bag, a wardrobe box, a jewelry box, a winde glass, a cordiale glass, a serving dish, a dessert dish, a milk cup, an espresso cup, a floor vase, a bud vase, a surf board, a cutting board, a walking stick, a cotton stick, a hot air balloon, a party balloon, a military tent, a camping tent, a traffic cone, an ice scream cone, a laundry bin, a desk bin"
