"""Norwegian (Bokmål) reading passages: natural prose, roughly 45-70 words, no digits.
Together they cover the retroflexes (rt, rd, rn, rl, rs), kj and skj, the rounded front
vowels (y, ø, u), æ, o versus å, and the diphthongs ei, øy and au."""

BANK = [
    "Hver lørdag går vi en lang tur i skogen bak huset. Barna løper foran og leter etter sopp, "
    "mens hunden snuser på hver eneste stein. Når vi kommer frem til vannet, tenner vi bål og "
    "griller pølser. Det er ikke alltid sol, men en kopp varm kakao gjør det meste bedre.",

    "Bestemor bodde i et lite gult hus ved fjorden. Om sommeren sto hun tidlig opp for å plukke "
    "bær i hagen, og om kvelden satt hun på trappen og strikket. Hun fortalte oss historier om "
    "den gangen hun rodde over fjorden alene i storm, og vi trodde på hvert eneste ord.",

    "Det kjente kjøpesenteret i sentrum skal stenge for godt neste høst. Mange kunder synes det er "
    "trist, særlig de eldre, som har handlet der hele livet. Eieren sier at folk heller kjøper "
    "klær og sko på nettet nå, og at de faste kostnadene har blitt altfor høye.",

    "Jeg glemte nøklene mine på kontoret i går, så jeg måtte ringe til naboen for å komme inn. "
    "Hun hadde heldigvis en ekstra nøkkel liggende i en skuff på kjøkkenet. Som takk bakte jeg "
    "en sjokoladekake til henne i dag, og hun ble så glad at hun inviterte meg på kaffe.",

    "Toget fra Oslo til Bergen er en av de vakreste jernbanestrekningene i verden. Først kjører "
    "man gjennom grønne daler og forbi blanke innsjøer, så klatrer toget opp på fjellet, der "
    "snøen ligger lenge utover våren. Mange turister sitter med nesen mot vinduet hele veien.",

    "Om vinteren går solen nesten ikke opp i Nord-Norge. Likevel er mørketiden ikke bare trist. "
    "Himmelen kan bli blå og rosa midt på dagen, og om natten danser nordlyset over fjellene. "
    "Folk tenner lys i alle vinduer, drikker mye kaffe og venter tålmodig på at solen skal komme tilbake.",

    "Min bror har nettopp kjøpt en gammel seilbåt som trenger mye arbeid. Hver helg står han i "
    "båthavna og skraper maling, skifter tau og fikser motoren. Kona hans tror at båten aldri "
    "kommer på sjøen, men han er sikker på at de skal seile langs kysten til sommeren.",

    "Skolen i bygda vår har bare tjue elever, fordelt på sju klassetrinn. De store hjelper de små "
    "med lesing og regning, og alle spiser lunsj sammen i det samme rommet. Læreren sier at det "
    "er den beste skolen hun har jobbet på, selv om den er nesten for liten.",

    "Kjøkkenet var fullt av damp da jeg kom hjem. Far sto ved komfyren og rørte i en stor gryte "
    "med fiskesuppe, mens radioen spilte gamle norske sanger. Han hadde kjøpt fersk torsk og reker "
    "på brygga om morgenen, og han lovet at suppen skulle bli den beste vi noen gang hadde smakt.",

    "Det norske været er kjent for å skifte raskt. En morgen kan begynne med sol og blå himmel, "
    "før det plutselig blåser opp og regner sidelengs. Derfor har de fleste nordmenn alltid en "
    "ekstra genser og en regnjakke i sekken, uansett hva værmeldingen sier.",

    "Vi var sju venner som leide en hytte på fjellet i påsken. Hver dag gikk vi lange turer på ski, "
    "og om kvelden spilte vi kort foran peisen. Den siste dagen kom det så mye snø at vi måtte "
    "grave oss ut av døra, men ingen av oss hadde lyst til å reise hjem.",

    "Biblioteket i byen vår har blitt et yrende møtested. Her kan man låne bøker, men også spille "
    "brettspill, sy på symaskin eller lære å programmere. Om ettermiddagen sitter ungdommer og gjør "
    "lekser ved de store vinduene, mens pensjonister leser aviser i de myke stolene ved kaffebaren.",

    "Kusinen min flyttet til Tromsø for å studere biologi. Hun forteller at hun aldri har sett så "
    "mye fugl som der oppe. Om våren kommer det tusenvis av sjøfugler til fuglefjellene langs "
    "kysten, og hun bruker mange timer med kikkert og notatbok for å telle dem.",

    "I går kveld hørte jeg en rar lyd fra hagen. Først trodde jeg at det var katten til naboen, "
    "men da jeg lyste med lommelykta, så jeg et lite pinnsvin som spiste av kattematen. Det så "
    "ikke redd ut i det hele tatt, og etter en stund ruslet det rolig inn under hekken.",

    "Når man lærer norsk, er det lett å blande sammen de lange og korte vokalene. Ordet tak er noe "
    "helt annet enn takk, og å hyle er ikke det samme som en hylle. Det hjelper å lytte nøye til "
    "hvordan folk snakker, og å lese høyt litt hver dag, gjerne de samme tekstene flere ganger.",

    "Gården ligger høyt over dalen, med utsikt mot breen. Om sommeren beiter kyrne og sauene på "
    "setra, og bonden går opp hver morgen for å telle dyrene. Datteren hans har lært å kjøre "
    "traktor, og hun sier at hun en dag skal overta hele gården sammen med broren sin.",

    "Etter middag liker vi å gå en kort tur langs elva. Vannet er klart og kaldt, og noen ganger "
    "ser vi ørret som står stille i strømmen. Sønnen min drømmer om å fiske en stor laks, så han "
    "har med seg fiskestanga hver gang, selv om han aldri har fått noe.",

    "Det første snøfallet kom tidlig i år. Barna i gata våknet til et hvitt landskap og løp ut i "
    "pysjamas for å lage snømann. Foreldrene måtte grave fram bilene og lete etter vintersko "
    "og votter i boden. Om ettermiddagen var hele bakken full av akende unger med røde kinn.",

    "Tante Kari driver et lite bakeri i en gammel trebygning ved torget. Hver morgen klokka fem "
    "begynner hun å bake brød, boller og skillingsboller. Duften sprer seg over hele torget, og "
    "lenge før hun åpner døra står det folk i kø utenfor, mange av dem med søvnige øyne.",

    "Fotballaget til datteren min vant den siste kampen i serien. De lå under hele første omgang, "
    "men etter pausen scoret de to mål på fem minutter. Treneren var så glad at han hoppet rundt "
    "på sidelinja, og etterpå spanderte han is på hele laget.",

    "Den gamle fyrvokteren bodde alene på en øy langt ute i havet. Han hadde ingen telefon, bare en "
    "radio og en liten robåt. Hver kveld tente han lyset i fyret, og hver morgen skrev han ned "
    "været i en tykk bok. Mange skip kom trygt i havn takket være ham.",

    "Jeg prøvde å lage norsk julemat for første gang i år. Ribba ble litt tørr, og surkålen var "
    "altfor søt, men alle gjestene var høflige og sa at det smakte nydelig. Neste år tror jeg at "
    "jeg ber svigermor om hjelp, eller kanskje bare bestiller mat fra restauranten på hjørnet.",

    "Fergen over fjorden går hver halvtime om sommeren. Turister står på dekket og tar bilder av "
    "de bratte fjellsidene og fossene som styrter ned i det grønne vannet. Mannskapet har sett det "
    "tusen ganger, men kapteinen sier at han fortsatt synes det er like vakkert hver eneste dag.",

    "Naboen vår er en pensjonert lærer som elsker å stelle hagen sin. Hun har roser, tulipaner og "
    "et lite drivhus med tomater og agurk. Om høsten deler hun ut epler og plommer til alle i "
    "gata, og hun blir skuffet hvis noen sier nei takk.",
]
