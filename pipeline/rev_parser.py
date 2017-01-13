# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:37:49 2016

@author: Rafael Crescenzi
"""


import re
import json
from collections import defaultdict

from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import fuzz
import unicodedata as ud
from scipy import stats

try:
    from langid import LanguageIdentifier, model
except:
    from langid.langid import LanguageIdentifier, model

import utils

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)


def xml_generator(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        rev_start = False
        rev = []
        for line in f:
            if rev_start:
                if "</revision>" in line:
                    rev_start = False
                    yield rev
                    rev = []
                else:
                    rev.append(line)
            elif "<revision>" in line:
                rev_start = True

tag_regex = re.compile(r'<(\w*)(>|.*>)')
text_regex = re.compile(r'>(.*)<')


def parse_xml(xml_text):
    cont_start = False
    revision = {}
    for i, line in enumerate(xml_text):
        if "<contributor>" in line:
            cont_start = True
        elif "<minor/>" in line:
            revision["minor"] = 1
        else:
            tag = tag_regex.search(line).group(1)
            if tag in ["id", "timestamp", "comment", "text"]:
                if cont_start:
                    tag = "user" + tag
                try:
                    revision[tag] = text_regex.search(line).group(1)
                except:
                    pass
            if "</contributor>" in line:
                cont_start = False
    return parse_revision(revision)

control_props = ["P227", "P213", "P244", "P245", "P214", "P268", "P269",
                 "P1025", "P270", "P271", "P349", "P396", "P409", "P1315",
                 "P502", "P503", "P496", "P497", "P508", "P640", "P646",
                 "P2671", "P691", "P709", "P718", "P723", "P724", "P727",
                 "P745", "P760", "P791", "P648", "P947", "P1005", "P1006",
                 "P1017", "P1043", "P1051", "P1052", "P1053", "P1054", "P1058",
                 "P1187", "P1188", "P1208", "P1209", "P1216", "P1217", "P1218",
                 "P1219", "P1220", "P1232", "P1233", "P1234", "P1235", "P1238",
                 "P1239", "P1243", "P1245", "P1250", "P1251", "P1252", "P1253",
                 "P1255", "P1270", "P1271", "P1272", "P1273", "P1274", "P1275",
                 "P1277", "P1280", "P1281", "P1284", "P1285", "P1286", "P1287",
                 "P1288", "P1289", "P1292", "P227", "P1293", "P1296", "P1297",
                 "P1309", "P1310", "P1310", "P1375", "P1385", "P1386", "P1391",
                 "P1392", "P1394", "P1395", "P1400", "P1415", "P1438", "P1439",
                 "P1447", "P1453", "P1461", "P1466", "P1601", "P1254", "P1263",
                 "P1291", "P1307", "P1331", "P1749", "P1803"]

def parse_revision(case):
    res = {
           "revisionid": int(case.get("id")),
           "userid": int(case.get("userid", -1)),
           "minor": int(case.get("minor", 0)),
           "timestamp":  case.get("timestamp", "").replace("T", " ").strip("Z")
    }

    if "comment" in case.keys():
        res.update({k: v for k, v in parse_comment(case["comment"]).items() if v})
    else:
        res["action"] = "emptyComment"


    base_json = """{
          "id": "-1",
          "type": "item",
          "labels": {},
          "descriptions": {},
          "aliases": {},
          "claims": {},
          "sitelinks": {}
      }"""

    obj_dict = case.get("text", base_json)
    res["json_len"] = len(obj_dict)
    obj_dict = json.loads(obj_dict)

    res["itemid"] = int(obj_dict.get("id", "-1").lower().strip("q"))

    try:
        res["instanceOf"] = obj_dict["claims"]["P31"][0]["mainsnak"]["datavalue"]["value"]["numeric-id"]
    except:
        res["instanceOf"] = None


    action = res.get("action", "unknown").lower().strip()
    subaction = res.get("subaction", "").lower().strip()


    if ("claim" in action) or ("qualifier" in action) or ("reference" in action):
        if "tail" in res.keys():
            res["tail"] = res["tail"].strip()
            res.update(parse_claim(res["tail"]))

    elif ("sitelink" in action) or ("label" in action + "_" + subaction) \
        or ("aliases" in action) or ("description" in action):
        if (action == "clientsitelink") and ("tail2" in res.keys()):
            temp = res["tail2"].split("|")
            res["lang"] = temp[0]
            res["tail"] = temp[-1].split(":")[-1]
        if "lang" in res.keys():
            res.update(parse_lang(res["lang"]))
        if "tail" in res.keys():
            res["tail"] = res["tail"].strip()
            res.update(parse_tail(res["tail"]))
        if ("lang" in res.keys()) and ("tail" in res.keys()) and ("text" in case.keys()):
            res.update(lang_probs(res["tail"], res["lang"]))
            if "sitelink" in action:
                tipo = "link"
            else:
                tipo = "label"
            res.update(label_link_similarity(tipo, res["lang"], res["tail"], obj_dict))
        if ("label" in action + "_" + subaction):
            res["afectedProperty"] = "label"
        elif ("aliases" in action):
            res["afectedProperty"] = "aliases"
        elif ("description" in action):
            res["afectedProperty"] = "description"

    elif (action == "wbeditentity"):
        if subaction == "create":
            res["afectedProperty"] = "item"
            res["action"] = "create_protect"
        else:
            res["afectedProperty"] = "unknown"
            res["action"] = "editentity"
            if "tail" in res.keys():
                res["tail"] = res["tail"].strip()
                res.update(parse_tail(res["tail"]))


    elif action in ["create_protect", "wbcreate", "special", "wbsetentity", "wblinktitles"]:
        res["afectedProperty"] = "item"
        res["action"] = "create_protect"


    elif action in ["undo", "undid", "restore", "revert"]:
        if action in ["undo", "undid"]:
            res["action"] = "undo"
        else:
            res["action"] = action
        res["afectedProperty"] = "unknown"
        if "prev_user" in res:
            if "bot" in res["prev_user"].lower():
                res["prev_user"] = "bot"
            elif res["prev_user"][0] in "0123456789":
                res["prev_user"] = "anonymous"
            else:
                res["prev_user"] = "registered"


    elif action in ["wbcreateredirect", "wbmergeitems"]:
        res["afectedProperty"] = "unknown"

    elif action == "emptycomment":
        res["afectedProperty"] = "item"

    else:
        if action not in ["removed", "undefined", "unknown"]:
            print("missed", action)
        res["action"] = "unknown"
        res["afectedProperty"] = "unknown"
        if "lang" in res.keys():
            res.update(parse_lang(res["lang"]))
        if "tail" in res.keys():
            res["tail"] = res["tail"].strip()
            res.update(parse_tail(res["tail"]))
        if ("lang" in res.keys()) and ("tail" in res.keys()) and ("text" in case.keys()):
            res.update(label_link_similarity("label", res["lang"], res["tail"], obj_dict))

    res["commentTailLength"] = len(res.get("tail", ""))

    if "tail" in res.keys():
        del res["tail"]
    if "tail2" in res.keys():
        del res["tail2"]

    return res


proper_regex = re.compile(r"\/\* (?P<action>\w*)(\-(?P<subaction>[\w\-]*))?:?(?P<amount>\d*)?(\|(?P<lang>[\w\-\_]*))?(\|{1,2}(?P<tail2>.*))? \*\/(?P<tail>.*)",
                          re.IGNORECASE)
fallback_regex = re.compile(r"\/\* (?P<action>\w*).*", re.IGNORECASE)
other_regex = re.compile(r'(?P<tail>.{0,})(?P<action>restore|revert|undo|undid).*(revision|edits).*\[\[special:contributions\/(?P<prev_user>.*)\|(?P=prev_user)\]\]',
                         re.IGNORECASE)


def parse_comment(comment):
    if comment.startswith("/*"):
        try:
            return proper_regex.search(comment).groupdict()
        except:
            return fallback_regex.search(comment).groupdict()
    else:
        l_comment = comment.lower()

    if ("created" in l_comment) or (("protect" in l_comment) and ("[[" in l_comment)):
        return {"action": "create_protect"}

    for act in ["revert", "undid", "restore", "undo"]:
        if (act in l_comment) and ("[[" in l_comment):
            try:
                return other_regex.search(comment).groupdict()
            except:
                return {"action": act, "tail": comment}
    return {"action": "unknown", "tail": comment}


claim_regex = re.compile(r"(?P<afectedProperty>p\d+)($|(\]\])?(: )?(\[\[q)?(?P<tail>.+)(\]\]))", re.IGNORECASE)

def parse_claim(tail):
    temp = claim_regex.search(tail)
    if temp:
        temp = {k: v for k, v in temp.groupdict().items() if v}
        if temp.get("tail") is not None:
            temp["tail"] = temp["tail"].strip()
            if temp["tail"].endswith("]]"):
                temp["tail"] = temp["tail"].strip("]]")
                temp["value_is_item"] = 1
                temp["is_authority_control"] = int(temp["tail"] in control_props)
            else:
                temp.update(parse_tail(tail))
        return temp
    else:
        return parse_tail(tail)


def lang_probs(tail, lang):
    if len(tail) == 0:
        return {}
    lang_2, prob = identifier.classify(tail)
    if lang == lang_2:
        return {"lang_prob": prob}
    else:
        return {"lang_prob": 1 - prob}

lang_regex = "(^|\\n)([ei]n )??(a(frikaa?ns|lbanian?|lemanha|ng(lais|ol)|ra?b(e?|[ei]c|ian?|isc?h)|rmenian?|ssamese|azeri|z[e\\u0259]rba(ijani?|ycan(ca)?|yjan)|\\u043d\\u0433\\u043b\\u0438\\u0439\\u0441\\u043a\\u0438\\u0439)|b(ahasa( (indonesia|jawa|malaysia|melayu))?|angla|as(k|qu)e|[aeo]ng[ao]?li|elarusian?|okm\\u00e5l|osanski|ra[sz]il(ian?)?|ritish( kannada)?|ulgarian?)|c(ebuano|hina|hinese( simplified)?|zech|roat([eo]|ian?)|atal[a\\u00e0]n?|\\u0440\\u043f\\u0441\\u043a\\u0438|antonese)|[c\\u010d](esky|e[s\\u0161]tina)\r\n|d(an(isc?h|sk)|e?uts?ch)|e(esti|ll[hi]nika|ng(els|le(ski|za)|lisc?h)|spa(g?[n\\u00f1]h?i?ol|nisc?h)|speranto|stonian|usk[ae]ra)|f(ilipino|innish|ran[c\\u00e7](ais|e|ez[ao])|ren[cs]h|arsi|rancese)|g(al(ego|ician)|uja?rati|ree(ce|k)|eorgian|erman[ay]?|ilaki)|h(ayeren|ebrew|indi|rvatski|ungar(y|ian))|i(celandic|ndian?|ndonesian?|ngl[e\\u00ea]se?|ngilizce|tali(ano?|en(isch)?))|ja(pan(ese)?|vanese)|k(a(nn?ada|zakh)|hmer|o(rean?|sova)|urd[i\\u00ee])|l(at(in[ao]?|vi(an?|e[s\\u0161]u))|ietuvi[u\\u0173]|ithuanian?)|m(a[ck]edon(ian?|ski)|agyar|alay(alam?|sian?)?|altese|andarin|arathi|elayu|ontenegro|ongol(ian?)|yanmar)|n(e(d|th)erlands?|epali|orw(ay|egian)|orsk( bokm[a\\u00e5]l)?|ynorsk)|o(landese|dia)|p(ashto|ersi?an?|ol(n?isc?h|ski)|or?tugu?[e\\u00ea]se?(( d[eo])? brasil(eiro)?| ?\\(brasil\\))?|unjabi)|r(om[a\\u00e2i]ni?[a\\u0103]n?|um(ano|\\u00e4nisch)|ussi([ao]n?|sch))|s(anskrit|erbian|imple english|inha?la|lov(ak(ian?)?|en\\u0161?[c\\u010d]ina|en(e|ij?an?)|uomi)|erbisch|pagnolo?|panisc?h|rbeska|rpski|venska|c?wedisc?h|hqip)|t(a(galog|mil)|elugu|hai(land)?|i[e\\u1ebf]ng vi[e\\u1ec7]t|[u\\u00fc]rk([c\\u00e7]e|isc?h|i\\u015f|ey))|u(rdu|zbek)|v(alencia(no?)?|ietnamese)|welsh|(\\u0430\\u043d\\u0433\\u043b\\u0438\\u0438\\u0441|[k\\u043a]\\u0430\\u043b\\u043c\\u044b\\u043a\\u0441|[k\\u043a]\\u0430\\u0437\\u0430\\u0445\\u0441|\\u043d\\u0435\\u043c\\u0435\\u0446|[p\\u0440]\\u0443\\u0441\\u0441|[y\\u0443]\\u0437\\u0431\\u0435\\u043a\\u0441)\\u043a\\u0438\\u0439( \\u044f\\u0437\\u044b\\u043a)??|\\u05e2\\u05d1\\u05e8\\u05d9\\u05ea|[k\\u043a\\u049b](\\u0430\\u0437\\u0430[\\u043a\\u049b]\\u0448\\u0430|\\u044b\\u0440\\u0433\\u044b\\u0437\\u0447\\u0430|\\u0438\\u0440\\u0438\\u043b\\u043b)|\\u0443\\u043a\\u0440\\u0430\\u0457\\u043d\\u0441\\u044c\\u043a(\\u0430|\\u043e\\u044e)|\\u0431(\\u0435\\u043b\\u0430\\u0440\\u0443\\u0441\\u043a\\u0430\\u044f|\\u044a\\u043b\\u0433\\u0430\\u0440\\u0441\\u043a\\u0438( \\u0435\\u0437\\u0438\\u043a)?)|\\u03b5\\u03bb\\u03bb[\\u03b7\\u03b9]\\u03bd\\u03b9\\u03ba(\\u03ac|\\u03b1)|\\u10e5\\u10d0\\u10e0\\u10d7\\u10e3\\u10da\\u10d8|\\u0939\\u093f\\u0928\\u094d\\u0926\\u0940|\\u0e44\\u0e17\\u0e22|[m\\u043c]\\u043e\\u043d\\u0433\\u043e\\u043b(\\u0438\\u0430)?|([c\\u0441]\\u0440\\u043f|[m\\u043c]\\u0430\\u043a\\u0435\\u0434\\u043e\\u043d)\\u0441\\u043a\\u0438|\\u0627\\u0644\\u0639\\u0631\\u0628\\u064a\\u0629|\\u65e5\\u672c\\u8a9e|\\ud55c\\uad6d(\\ub9d0|\\uc5b4)|\\u200c\\u0939\\u093f\\u0928\\u0926\\u093c\\u093f|\\u09ac\\u09be\\u0982\\u09b2\\u09be|\\u0a2a\\u0a70\\u0a1c\\u0a3e\\u0a2c\\u0a40|\\u092e\\u0930\\u093e\\u0920\\u0940|\\u0c95\\u0ca8\\u0ccd\\u0ca8\\u0ca1|\\u0627\\u064f\\u0631\\u062f\\u064f\\u0648|\\u0ba4\\u0bae\\u0bbf\\u0bb4\\u0bcd|\\u0c24\\u0c46\\u0c32\\u0c41\\u0c17\\u0c41|\\u0a97\\u0ac1\\u0a9c\\u0ab0\\u0abe\\u0aa4\\u0ac0|\\u0641\\u0627\\u0631\\u0633\\u06cc|\\u067e\\u0627\\u0631\\u0633\\u06cc|\\u0d2e\\u0d32\\u0d2f\\u0d3e\\u0d33\\u0d02|\\u067e\\u069a\\u062a\\u0648|\\u1019\\u103c\\u1014\\u103a\\u1019\\u102c\\u1018\\u102c\\u101e\\u102c|\\u4e2d\\u6587(\\u7b80\\u4f53|\\u7e41\\u9ad4)?|\\u4e2d\\u6587\\uff08(\\u7b80\\u4f53?|\\u7e41\\u9ad4)\\uff09|\\u7b80\\u4f53|\\u7e41\\u9ad4)( language)??($|\\n)"

lang_regex = re.compile(lang_regex, re.IGNORECASE)
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url_regex = re.compile(url_regex, re.IGNORECASE)

def parse_tail(tail):
    tail_words =  RegexpTokenizer(r'\w+').tokenize(tail)
    tail_length = len(tail_words)

    if tail_length == 0:
        return {}

    word_feats = {
        "lowerCaseWordRatio": 0,
        "upperCaseWordRatio": 0,
        "badWordRatio": 0,
        "languageWordRatio": 0,
    }

    long_w = 0
    for word in tail_words:
        lw = len(word)
        if lw == 0:
            continue
        elif lw > long_w:
            long_w = lw
        if lang_regex.search(word):
            word_feats["languageWordRatio"] += 1
        elif word.lower() in utils.bad_words:
            word_feats["badWordRatio"] += 1
        if word[0].islower():
            word_feats["lowerCaseWordRatio"] += 1
        elif word[0].isupper():
            word_feats["upperCaseWordRatio"] += 1

    word_feats = {k: word_feats[k] / tail_length for k in word_feats.keys()}
    word_feats["containsLanguageWord"] = int(word_feats["languageWordRatio"] > 0)
    word_feats["longestWord"] = long_w

    char_feats = {
        "upperCaseRatio": 0,
        "lowerCaseRatio": 0,
        "simbolRatio": 0,
        "alphanumericRatio": 0,
        "digitRatio": 0,
        "punctuationRatio": 0,
        "bracketRatio": 0,
        "whitespaceRatio": 0,
        "latinRatio": 0,
        "nonLatinRatio": 0
    }

    alphabets = []
    prev_char = ""
    char_seq = [1]
    for t in tail:
        if t == prev_char:
            char_seq[-1] += 1
        else:
            char_seq.append(1)
        prev_char = t
        if t == " ":
            char_feats["whitespaceRatio"] += 1
        elif t in ",.;:´¨'\"?¿!¡":
            char_feats["punctuationRatio"] += 1
        elif t in "&%$#@+-_*/\\":
            char_feats["simbolRatio"] += 1
        elif t in "{}[]()":
            char_feats["bracketRatio"] += 1
        elif t.isupper():
            char_feats["upperCaseRatio"] += 1
        elif t.islower():
            char_feats["lowerCaseRatio"] += 1
            char_feats["alphanumericRatio"] += 1
        elif t in "1234567890":
            char_feats["digitRatio"] += 1
            char_feats["alphanumericRatio"] += 1
        alphabets.append(ud.name(t, "unknown").split(" ")[0])
        if alphabets[-1] == "LATIN":
            char_feats["latinRatio"] += 1
        else:
            char_feats["nonLatinRatio"] += 1
    tl = len(tail)
    char_feats = {k: char_feats[k] / tl for k in char_feats.keys()}
    char_feats["main_alphabet"] = stats.mode(alphabets)[0][0]
    char_feats["longestCharacterSequence"] = max(char_seq)

    char_feats.update(word_feats)
    if url_regex.search(tail):
        char_feats["containsUrl"] = 1
    else:
        char_feats["containsUrl"] = 0
    return char_feats


def parse_lang(lang):
    res = {}
    if "wiki" in lang:
        if lang in ["commonswiki", "simplewiki", "specieswiki"]:
            res["afectedProperty"] = lang
        else:
            index = lang.index("wiki")
            res["lang"], res["afectedProperty"] = lang[:index], lang[index:]
    else:
        res["lang"] = lang
    if "lang" in res.keys():
        locale_lang_char = [c for c in ["_", "-"] if c in res["lang"]]
        if len(locale_lang_char) > 0:
            ar = res["lang"].split(locale_lang_char[0])
            res["lang"], res["lang_locale"] = ar[0], locale_lang_char[0].join(ar[1:])
    return res


def label_link_similarity(tipo, lang, tail, obj_dict):
    if len(tail) == 0:
        return {}

    if tipo == "link":
        labels = obj_dict.get("labels", {})
        if type(labels) == list:
            return {}
        else:
            compare = [labels[k].get("value", "") for k in labels.keys() if lang in k]
            if len(compare) == 0:
                compare = [labels[k].get("value", "") for k in labels.keys()]
            if len(compare) == 0:
                return {}
    elif tipo == "label":
        links = obj_dict.get("sitelinks", {})
        if type(links) == list:
            return {}
        else:
            compare = [links[k].get("title", "") for k in links.keys() if lang in k]
            if len(compare) == 0:
                compare = [links[k].get("title", "") for k in links.keys()]
            if len(compare) == 0:
                return {}

    return compute_similarities(tail, compare)


def compute_similarities(tail, arr):
    res = defaultdict(list)
    for a in arr:
        try:
            res["fuzzy_total"].append(fuzz.ratio(tail, a))
        except:
            pass
        try:
            res["fuzzy_partial"].append(fuzz.partial_ratio(tail, a))
        except:
            pass
    try:
        res = {k: max(res[k]) for k in res.keys()}
    except:
        res = {}
    return res
