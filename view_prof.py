import pstats
p = pstats.Stats("foostats")
p.sort_stats('time').print_stats(10)
