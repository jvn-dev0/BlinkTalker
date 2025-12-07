# Module 4 – Morse Decoder

MORSE_MAP = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "/": " ",
}


class MorseDecoder:
    def __init__(self):
        self.buffer = ""  # current letter in morse
        self.current_word = ""  # decoded full text

    def add_symbol(self, s):
        self.buffer += s

    def end_letter(self):
        if not self.buffer:
            return ""
        letter = MORSE_MAP.get(self.buffer, "?")
        self.current_word += letter
        self.buffer = ""
        return letter

    def end_word(self):
        self.current_word += " "

    def get_text(self):
        return self.current_word
