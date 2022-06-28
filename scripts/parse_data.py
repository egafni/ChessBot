import argparse
import chess
import numpy
import bz2
import tqdm
from chess.pgn import Game

def parse_game(game: Game):
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
    
    winner_int = 2-int(headers['Result'][-1]) #2 -> w, 1 -> b, 0 -> draw
    # winner = {2:'W', 1:'B', 0:'D'}[winner_int]
    winner = winner_int

    if ((min(elo) < 1510) or (t < 300)) and (min(elo) < 2000):
        return False
    elif (len(uci) < 10) or (len(uci) > 200):
        return False

    rank = 'H' if min(elo) >= 2000 and (t >= 300) else 'L'
    
    uci.append(winner)
    
    return rank + ' ' + ' '.join(uci)

def main(args):
    games = []
    
    with bz2.open(args.input_path, 'rt') as fp:
        for _ in tqdm.tqdm(iter(int, 1)): # infinite loop
            game = chess.pgn.read_game(fp)
            if game:
                game_str = parse_game(game)
                games.append(game_str)
                
            if args.max_games and len(games) >= args.max_games:
                break
            
    numpy.save(args.input_path+'.npy', numpy.array(games))
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_path')
    p.add_argument('-o', '--output_npz')
    p.add_argument('-m', '--max_games', type=int)
    
    args = p.parse_args()
    main(args)

    """        
    python scripts/parse_data.py \
        -i /static/chess/data/lichess_db_standard_rated_2014-11.pgn.bz2 \
        -o  /static/chess/data/lichess_db_standard_rated_2014-11.pgn.npy
    """