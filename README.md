# Beat and Downbeat Tracking
ECS7006 Music Informatics 2021, Coursework 2

A Shazam-style audio fingerprinting system.

### Dependencies
```
  pip install -r requirements.txt
```

### Usage

``` python
from main import fingerprintBuilder, audioIdentification
# Building the fingerprinting database
fingerprintBuilder(database_path, fingerprint_path)
# Search for the best match
audioIdentification(query_path, fingerprint_path, output_file)

```

Please refer to __examples.ipynb__ for detailed examples.

### References
[1] Avery Wang. An industrial strength audio search algorithm. In ISMIR, pages 7-13, 2003.

[2] Peng Li, Songsheng Pan, and Huaping Liu. "THE 2020 NETEASE AUDIO FINGERPRINT SYSTEM." 2020.

[3] https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html

[4] https://github.com/leonardltk/Shazam-An-Industrial-Strength-Audio-Search-Algorithm-