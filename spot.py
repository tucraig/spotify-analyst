'''
Spotify Music Interpreter:
    Identifying my music taste with Linear Algebra
Version 1.1

Author: Tucker Craig
Date: 10/18/2019
'''
import sys, os, csv, math

import util
from tqdm import tqdm
import requests

import pandas as pd
import numpy as np
from numpy import linalg

from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
#import simplejson as json
#import spotipy

'''
Hook into the Spotify API
Returns the user's token
'''
def hook():
    scope = 'user-library-read, playlist-read-private'
    uname = os.getenv('SPOTIFY_USERNAME')
    utoken = os.getenv('USER_TOKEN')
    token = util.prompt_for_user_token(uname, scope)
    return token

'''
Helper function for scraping each song from the user's library
Returns the blob of songs from query
'''
def read_from_lib(user_token, limit, offset):
    url = "https://api.spotify.com/v1/me/tracks?limit={}&offset={}".format(limit, offset)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + user_token
    }
    spot_blob = requests.get(url, headers=headers).json()
    return spot_blob

def read_from_playlist(user_token, playlist_id, limit, offset):
    url = "https://api.spotify.com/v1/playlists/{}/tracks?limit={}&offset={}".format(playlist_id, limit, offset)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + user_token
    }
    spot_blob = requests.get(url, headers=headers).json()
    return spot_blob

'''
Make a call to the Spotify API to pick up songs
Exports the track urls and date added to a csv
'''
def get_urls(user_token, playlist_id=None):
    if (playlist_id):
        spot_blob = read_from_playlist(user_token,playlist_id,1,0)
    else:
        spot_blob = read_from_lib(user_token,1,0)
    total = spot_blob['total']
    write_started = False

    #for i in range(math.floor(total / 10)):
    for i, num in enumerate(tqdm(range(math.ceil(total / 10)))):
        offset = i * 10
        temp_lib = {}

        # curl at this offset
        if (playlist_id):
            spot_blob = read_from_playlist(user_token,playlist_id,10,offset)
        else:
            spot_blob = read_from_lib(user_token,10,offset)
        # dump this information into an array, or a csv
        song_list = spot_blob['items']
        
        temp_lib_p1 = dict([i['track']['id'],i['added_at']] for i in song_list) # 
        temp_lib_p2 = dict([i['track']['id'],i['track']['name']] for i in song_list)

        temp_feats = get_features(user_token,','.join(temp_lib_p1.keys()))
        for i in temp_feats:
            i['added_at'] = temp_lib_p1[i['id']]
            i['name'] = temp_lib_p2[i['id']]
            i.pop('type',None)
            i.pop('uri',None)
            i.pop('track_href',None)
            i.pop('analysis_url',None)
            
            temp_lib[i['id']] = i
        
        if (write_started):
            with open('user_library.csv', 'a', newline='\n', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(temp_lib.values())[0].keys())
                for data in temp_lib.values():
                    writer.writerow(data)
        else:
            with open('user_library.csv', 'w', newline='\n', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(temp_lib.values())[0].keys())
                writer.writeheader()
                for data in temp_lib.values():
                    writer.writerow(data)
            write_started = True
    return

'''
Extrapolate features from track urls in csv
Exports the track features to a csv
'''
def get_features(user_token, ids):
    url = "https://api.spotify.com/v1/audio-features/?ids=" + ids
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + user_token
    }
    spot_blob = requests.get(url, headers=headers).json()['audio_features']
    return spot_blob

'''
Find the distance between 2-d array and every entry of matrix
Return a new matrix, with entries being the distance to the array
'''
def euclid_dist(t1, t2):
    return np.sqrt(((t1-t2)**2).sum(axis = 1))

'''
Find the average song, then find song closest to this average
Return the name of the average song
'''
def analyze(filename='user_library.csv'):
    # importing data
    df = pd.read_csv(filename).drop(columns=['duration_ms','tempo'])
    print(df.describe())
    temp_df = df.drop(columns=['id', 'name', 'added_at'])

    # normalizing the values
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(temp_df)
    df_normalized = pd.DataFrame(scaled_array)

    # finding minimum average distance to each song
    D1 = pdist(df_normalized, 'euclidean')
    Z1 = squareform(D1)
    Z1_mean = pd.DataFrame(Z1).mean()

    test_dist = np.where(Z1_mean == np.amin(Z1_mean))[0][0]
    test_song = df.values[test_dist]
    test_song_name = test_song[-1]
    
    test_far_dist = np.where(Z1_mean == np.amax(Z1_mean))[0][0]
    test_far_song = df.values[test_far_dist]
    test_far_song_name = test_far_song[-1]
    print ("Closest to every song:\nClosest: {}, {}\nFurthest: {}, {}\n".format(test_song_name, np.amin(Z1_mean), test_far_song_name, np.amax(Z1_mean)))

    for i in range(4):
        df = df[df.name != test_song_name]
        df = df[df.name != test_far_song_name]
        temp_df = df.drop(columns=['id', 'name', 'added_at'])

        # normalizing the values
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_array = min_max_scaler.fit_transform(temp_df)
        df_normalized = pd.DataFrame(scaled_array)

        # finding minimum average distance to each song
        D1 = pdist(df_normalized, 'euclidean')
        Z1 = squareform(D1)
        Z1_mean = pd.DataFrame(Z1).mean()

        test_dist = np.where(Z1_mean == np.amin(Z1_mean))[0][0]
        test_song = df.values[test_dist]
        test_song_name = test_song[-1]
        
        test_far_dist = np.where(Z1_mean == np.amax(Z1_mean))[0][0]
        test_far_song = df.values[test_far_dist]
        test_far_song_name = test_far_song[-1]
        print(test_song_name)

    print ('\n--------\nClosest to average:')
    df = pd.read_csv(filename).drop(columns=['duration_ms','tempo'])

    # taking the average of this normalized dataframe
    avg = df_normalized.mean()

    # finding how close each entry is to the average
    dist = euclid_dist(avg, df_normalized) # as opposed to avg, temp_df
    
    # the song that's the closest to the average
    my_dist = np.where(dist == np.amin(dist))[0][0]
    my_song = df.values[my_dist]
    my_song_name = my_song[-1]

    # print(my_song)
    # print(df_normalized.values[my_dist])
    # print(df_normalized.values[1:5])
    
    # the song that's the furthest from the average
    far_dist = np.where(dist == np.amax(dist))[0][0]
    far_song = df.values[far_dist]
    far_song_name = far_song[-1]

    closest = my_song_name
    c_dist = np.amin(dist)
    furthest = far_song_name
    f_dist = np.amax(dist)
    print("{}\n(1): {}, {}\n".format(df_normalized.describe(), my_song_name, np.amin(dist)))

    for i in range(4):
        df = df[df.name != my_song_name]
        df = df[df.name != far_song_name]
        temp_df = df.drop(columns=['id', 'name', 'added_at'])
        # normalizing the values
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_array = min_max_scaler.fit_transform(temp_df)
        df_normalized = pd.DataFrame(scaled_array)
         # taking the average of this normalized dataframe
        avg = df_normalized.mean()

        # finding how close each entry is to the average
        dist = euclid_dist(avg, df_normalized) # as opposed to avg, temp_df
        
        # the song that's the closest to the average
        my_dist = np.where(dist == np.amin(dist))[0][0]
        my_song = df.values[my_dist]
        my_song_name = my_song[-1]

        # the song that's the furthest from the average
        far_dist = np.where(dist == np.amax(dist))[0][0]
        far_song = df.values[far_dist]
        far_song_name = far_song[-1]

        print(far_song_name)
        print("{}\n({}): {}, {}\n".format(df_normalized.describe(), i+2, my_song_name, np.amin(dist)))

    return ("Closest: {}, {}\nFurthest: {}, {}\n".format(closest, c_dist, furthest, f_dist))

'''
Wrapper for hooking, gathering, and applying algorithm
Arguments: get_data - boolean, defining whether the downloaded data should be refreshed
Returns 0 on clean exit
'''
def main(get_data=True):
    if(get_data):
        user_token = hook()
        #get_urls(user_token)
        get_urls(user_token, '4GLMvYLD7N9TcQG9GgQG69')
    print(analyze())
    return 0

if __name__ == '__main__':
    # if (not (os.getenv('SPOTIFY_USERNAME') and os.getenv('USER_TOKEN'))):
    #     print("\nMake sure you set the user name and token:\n\n"
    #         "  In the same path as this file, make a file '.env' and copy/set the following:\n\n"
    #         "------------\nSPOTIFY_USERNAME = {name}\n"
    #         "USER_TOKEN = {token}\n\n------------")
    # else:
    #     print("expected ")
    if (len(sys.argv) > 1):
        try:
            opt = int(sys.argv[1])
        except Exception as e:
            print("Expected option 1 or 0, if script should update songslist.")
            exit(-1)
        if opt in [0,1]:
            main(bool(opt))
        else:
            print("Expected option 1 or 0, if script should update songslist.")
    else:
        main(False)