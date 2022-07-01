import argparse
import chess
import numpy
import bz2
import tqdm
from chess.pgn import Game

def parse_game(game: Game, min_moves=10, max_moves=200):
    """Parses a chess.pgn.Game into a single string like:
    
    Args:
        game (Game): a chess Game object
        min_moves (int, optional): minimum number of moves to keep game. Defaults to 10.
        max_moves (int, optional): maximum number of moves to keep game. Defaults to 200.

    Returns:
        A UCI string of the game, ex 'B e2e4 b8c6 W'
    """
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
    winner = {2:'W', 1:'B', 0:'D'}[winner_int]
    # winner = winner_int

    if ((min(elo) < 1510) or (t < 300)) and (min(elo) < 2000):
        return False
    elif (len(uci) < min_moves) or (len(uci) > max_moves):
        return False

    rank = 'H' if min(elo) >= 2000 and (t >= 300) else 'L'
    
    uci.append(winner)
    
    return rank + ' ' + ' '.join(uci)

def main(args):
    games = []
    
    def save():
        numpy.save(args.input_path+'.npy', numpy.array(games))
    
    with bz2.open(args.input_path, 'rt') as fp:
        for _ in tqdm.tqdm(iter(int, 1)): # infinite loop
            game = chess.pgn.read_game(fp)
            if game:
                game_str = parse_game(game)
                if game_str:
                    games.append(game_str)
                else:
                    # parse_game returned False
                    pass
            else:
                break

            if len(games) % 10000 == 0:
                # save file every 10k games
                save()
                
            if args.max_games and len(games) >= args.max_games:
                break
            
    save()
    
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