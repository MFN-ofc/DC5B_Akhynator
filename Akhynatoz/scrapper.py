import requests
from bs4 import BeautifulSoup
import pandas as pd

urls = [
    'https://www.imdb.com/search/title/?genres=action&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWHAPZB277C051JR1R5&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_1',
    'https://www.imdb.com/search/title?genres=adventure&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=BNVGVA2ANC8ZCQJSBW5D&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_2',
    'https://www.imdb.com/search/title/?genres=animation&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_3',
    'https://www.imdb.com/search/title?genres=biography&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_4',
    'https://www.imdb.com/search/title?genres=comedy&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_5',
    'https://www.imdb.com/search/title?genres=crime&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_6',
    'https://www.imdb.com/search/title?genres=drama&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_7',
    'https://www.imdb.com/search/title?genres=family&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_8',
    'https://www.imdb.com/search/title?genres=fantasy&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_9',
    'https://www.imdb.com/search/title?genres=film_noir&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_10',
    'https://www.imdb.com/search/title?genres=history&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_11',
    'https://www.imdb.com/search/title?genres=horror&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_12',
    'https://www.imdb.com/search/title?genres=music&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_13',
    'https://www.imdb.com/search/title?genres=musical&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_14',
    'https://www.imdb.com/search/title?genres=mystery&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_15',
    'https://www.imdb.com/search/title?genres=romance&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_16',
    'https://www.imdb.com/search/title?genres=sci_fi&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_17',
    'https://www.imdb.com/search/title?genres=sport&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_18',
    'https://www.imdb.com/search/title?genres=thriller&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_19',
    'https://www.imdb.com/search/title?genres=war&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_20',
    'https://www.imdb.com/search/title?genres=western&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=94365f40-17a1-4450-9ea8-01159990ef7f&pf_rd_r=9XWF06KD4CN3N937TZRA&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_21'
]

filtered_data = []
id = 0
s = requests.Session()
cookies = {}

headers = {
    "Host": "www.imdb.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/112.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.imdb.com/chart/top/?ref_=nv_mv_250",
    "Connection": "keep-alive",
    "Cookie": "session-id=134-3314250-0428143; session-id-time=2082787201l; csm-hit=tb:8Y18NKF7724HH0RCPJC2+s-8Y18NKF7724HH0RCPJC2|1682424685060&t:1682424685060&adb:adblk_yes; uu=eyJpZCI6InV1NjJkOWQxZTEyYWU2NDQ0OGI2NmUiLCJwcmVmZXJlbmNlcyI6eyJmaW5kX2luY2x1ZGVfYWR1bHQiOmZhbHNlfX0=; ubid-main=134-0501809-5612660; session-token=HwKL2FH8Dnb2ILQyPo521fJgbN9irhc4X5dPL6uOwORS0DHQYneFDtzCgMhpfRSH2I1urDrwNFX+h5ABlishF3vDr07oSxgnEpSms+xFroi5rtyG7hhMmgUdrAANRTD/2vmTUvLvhn5O5BxAdn1nJagUM61O8ze8EZHxSbmnsYkt/YY+PQOtG5DtGev7RuckihGXIXmYtUILdxReifb2DaRlfbMkyNkPLrvnj0sUTzM=",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "TE": "trailers"
}

for url in urls:
    r = s.get(url, headers=headers, cookies=cookies)
    soup = BeautifulSoup(r.text, "html.parser")
    films = soup.find_all('div', {'class': 'lister-item'})
    for film in films:
        div_name = film.find('h3', {'class': 'lister-item-header'})
        name = div_name.find('a').text.strip()
        year = div_name.find('span', {'class': 'lister-item-year'}).text.strip()

        if film.find('span', {'class': 'certificate'}) is not None:
            certificate = film.find('span', {'class': 'certificate'}).text.strip()
        else:
            certificate = ""

        runtime = film.find('span', {'class': 'runtime'}).text.strip()
        genre = film.find('span', {'class': 'genre'}).text.strip()
        note = film.find('div', {'class': 'ratings-imdb-rating'}).text.strip()

        sort_num = film.find('p', {'class': 'sort-num_votes-visible'})
        spans = sort_num.find_all('span', {'name': 'nv'})
        if len(spans) >= 2:
            vote = spans[0].text.strip()
            gross = spans[1].text.strip()
        else:
            vote = None
            gross = None

        filtered_data.append([id, name, year, certificate, runtime, genre, note, vote, gross])
        id += 1

df = pd.DataFrame(filtered_data, columns=['ID', 'Nom', 'Année de sortie', 'Classification', 'Durée', 'Genre', 'Note', 'Nombre de votant', 'Revenus'])
df.to_csv('/Users/theorobert/Desktop/dc5b_clean_robert_theo_scrapping_dmm/scrapping akinatorzer/result.csv', index=False)
print("Fichier CSV a été créé avec douceur")
