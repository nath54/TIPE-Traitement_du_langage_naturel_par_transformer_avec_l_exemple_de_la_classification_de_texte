from mastodon import Mastodon

Mastodon.create_app(
    'test1',
    api_base_url='https://mastodon.social',  # Replace with your instance
    to_file='clientcred.secret'
)


# Log in - either every time, or use persisted
mastodon = Mastodon(
    client_id = "...",
    client_secret= "...",
    access_token= "...",
    api_base_url = 'https://mastodon.social'  # Change this to the instance you want to connect to
)
mastodon.log_in(
    "...",  # Change this to your email
    "...",  # Change this to your password
    to_file = 'pytooter_usercred.secret'
)

# Retrieve toots with a specific hashtag
toots = mastodon.timeline_hashtag('#city', limit=100)

for toot in toots:
    print(toot['content'])
