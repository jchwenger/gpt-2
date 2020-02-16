String text = "";
String tempText = "";
String[] lines = new String[0];
String[] newLines = new String[0];

void updateText(String newText) {
  text = newText;
  tempText = "";
}


boolean hasChangedAnswer() {
  if (newLines == null) return false;
  if (newLines.length != lines.length) {
    return true;
  }
  for (int i = 0; i < lines.length; i++) {
    if (newLines[i].hashCode() != lines[i].hashCode()) return true;
  }
  return false;
}

void updateLines() {
  newLines = loadStrings("data/answer.txt");
  if (hasChangedAnswer()) {
    lines = newLines;
    updateText(join(lines, " "));
  }
}

void nextChar() {
  int ttl = tempText.length(); 
  int tl = text.length();
  if (ttl < tl) { 
    tempText = text.substring(0, ttl + 1);
  }
}

void setup() {
  fullScreen();
  textFont(loadFont("TimesNewRomanPSMT-48.vlw"));
}

void draw() {
  background(255);
  fill(0);
  text(tempText, 32, 32, width - 64, height - 64);
  if (frameCount % 4 == 0) nextChar();
  if (frameCount % 100 == 0) updateLines();
}
