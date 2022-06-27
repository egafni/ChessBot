from chess import pgn

def read_game(game: pgn.Game):
    headers = game.headers
    time_control = headers['TimeControl']

    if headers['Termination'] == 'Unterminated' or headers['Result'] == '*':
        return False
    
    if '+' in time_control:
        t0, t1 = time_control.split('+')
        t = int(t0) + 40 * int(t1)
    elif time_control == '-':
        t = -1

    try:
        elo = (int(headers['WhiteElo']), int(headers['BlackElo']))
    except:
        elo = (0,0)
        
    uci = [x.uci() for x in game.mainline_moves()]
    res = 2-int(headers['Result'][-1]) #2 -> w, 1 -> b, 0 -> tie

    if ((min(elo) < 1510) or (t < 300)) and (min(elo) < 2000):
        return False
    elif (len(uci) < 10) or (len(uci) > 200):
        return False

    rank = 'A' if min(elo) >= 2000 and (t >= 300) else 'B'
    
    uci.append(str(res))
    
    return rank + ' ' + ' '.join(uci)

