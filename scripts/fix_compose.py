p = r'C:\Users\jeffj\open-brain\docker-compose.yml'
text = open(p, encoding='utf-8').read()
fixed = text.replace(':***@', ':something@')
open(p, 'w', encoding='utf-8').write(fixed)
