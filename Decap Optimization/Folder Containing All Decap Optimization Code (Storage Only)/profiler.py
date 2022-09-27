import pstats

p = pstats.Stats('GA_Profile')
p.sort_stats('tottime').print_stats(1000)