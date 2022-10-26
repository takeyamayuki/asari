from asari.api import Sonar
sonar = Sonar()
text = "表情筋が衰えてきてる。まずいな…"


print(sonar.ping(text=text))
