# 我艰难地结束了我的 Spotify 历史。

> 原文：<https://towardsdatascience.com/i-wrapped-my-spotify-history-the-hard-way-93dc832d9b47?source=collection_archive---------12----------------------->

## **用 Go 和 PostgreSQL 分析一年的流。**

![](img/1664941636258a3284a5d5f65c3bb268.png)

到目前为止，各地的 Spotify 用户已经获得了他们的 [2019 年包装](https:/spotify.com/wrapped/)结果。我对自己的经历并不感到惊讶，部分原因是我知道自己花了很多时间在一支 90 年代即将回归的乐队[的音乐中锻炼，还因为我最近使用了 Spotify 的](https://www.forbes.com/sites/stevebaltin/2019/11/01/confirmed-rage-against-the-machine-to-reunite-in-2020-headline-coachella/#5d56860812b1)[隐私数据下载](https://www.makeuseof.com/tag/download-privacy-data-spotify/)来直接查看我过去一年的历史，Spotify 保留了所有细节。

# 从 Spotify 获取我的播放历史

它始于在[https://www.spotify.com/ca-en/account/privacy/](https://www.spotify.com/ca-en/account/privacy/)的一个请求，在那里(靠近底部)，Spotify 用户可以请求下载，包括“你的播放列表副本、搜索、过去一年的流媒体历史、你的库中保存的项目列表、你的粉丝数量、你关注的其他用户和艺术家的数量和姓名，以及你的付费和订阅数据。”

那似乎是很多！网站上说最多需要 30 天([按照 GDPR 的要求](https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/individual-rights/right-of-access/))，但是根据我的经验(两次请求)，收到一封包含链接的邮件需要三四天。该链接指向一个可使用两周的 zip 文件。解压存档文件时，我找到了这些文件:

```
$ ls -s
total 4428
   4  CarThing.json         16  SearchQueries.json
   4  FamilyPlan.json     1404  StreamingHistory0.json
   4  Follow.json         1412  StreamingHistory1.json
   0  KidsAccount.json     980  StreamingHistory2.json
   4  Payments.json          4  Userdata.json
  92  Playlist1.json       160  YourLibrary.json
 344 'Read Me First.pdf'
```

关于 Spotify 如何使用它的详细信息在 Spotify 的[隐私中心](https://www.spotify.com/us/privacy/)中，但是文件名大部分是不言自明的。你会看到信用卡的详细信息被屏蔽了，我们没有得到任何我们关注的帐户的身份，但如果你想知道你的收听历史，它都在这里！查看名为`StreamingHistory[0-2].json`的文件，我发现每个文件有多达 10，000 条记录，每条记录描述了一个流媒体播放，包括其时间和持续时间。比如 2018 年 11 月 22 日，我正在怀旧地听 [Rush](https://www.rush.com/band/) :

```
{
    "endTime" : "2018-11-22 20:57",
    "artistName" : "Rush",
    "trackName" : "Subdivisions",
    "msPlayed" : 334722
},
```

部分播放包括在内，流可能短至 0 毫秒，我猜这相当于快速跳过播放列表。

你可能会惊讶地发现(就像我一样),这不包括艺术家和歌曲的 Spotify URIs。例如，[分部](https://open.spotify.com/track/0Z0s6dw0zw2ENU1gVjlLV6?si=UzTRYEAHRyCl_wmpkG-HlQ)的 Spotify URI 是`spotify:track:0Z0s6dw0zw2ENU1gVjlLV6`，而 [Rush](https://open.spotify.com/artist/2Hkut4rAAyrQxRdof7FVJq?si=67GmcS_USUW7k-j8hWjwNQ) 的 URI 是`spotify:artist:2Hkut4rAAyrQxRdof7FVJq`。但这可能只是意味着这些数据更多的是给人类使用，而不是给机器使用。

我决定用 Go 来分析我的流历史，并从编写一些代码来读取我的一个流历史文件的内容开始:

```
package mainimport (
    “encoding/json”
    “fmt”
    “io/ioutil”
)type Play struct {
    EndTime string `json:”endTime”`
    ArtistName string `json:”artistTime”`
    TrackName string `json:”trackName”`
    MSPlayed int32 `json:”msPlayed”`
}func main() {
    b, err := ioutil.ReadFile(“StreamingHistory0.json”)
    if err != nil {
        panic(err)
    }
    fmt.Printf(“%d\n”, len(b))
    var plays []Play
    json.Unmarshal(b, &plays)
    fmt.Printf(“%d\n”, len(plays))
}
```

# 将我的游戏历史加载到 PostgreSQL

我可以直接在 Go 中完成剩下的工作，但是因为我也想温习一下 PostgreSQL，所以我决定将这些记录加载到本地数据库中，用 SQL 对它们进行分析。

因为我已经在我的桌面 Ubuntu 机器上安装了 PostgreSQL(这里的指令[是](https://www.postgresql.org/download/linux/ubuntu/))，所以我的第一步是为数据设计一个模式。我决定将它正常化，将艺术家、曲目和流式播放分离到单独的表中。

```
CREATE TABLE artists (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);CREATE TABLE tracks (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  artist_id INTEGER NOT NULL REFERENCES artists (id)
);CREATE TABLE streaming_plays (
  id SERIAL PRIMARY KEY,
  end_time TIMESTAMPTZ NOT NULL,
  ms_played INTEGER NOT NULL,
  track_id INTEGER NOT NULL REFERENCES tracks (id)
);
```

我将它放在一个文件中，并从命令行直接加载到 PostgreSQL 中(首先用`createdb playtime`创建了`playtime`数据库之后):

```
psql -U postgres -d playtime -a -f schema.sql
```

接下来，我需要一种将数据从 Go 加载到 PostgreSQL 的方法。在浏览各种选项时，我发现了 [gnorm](https://github.com/gnormal/gnorm) 和 [postgres-go](https://github.com/gnormal/postgres-go) ，它们直接从数据库模式生成 go 支持代码。 [postgres-go](https://github.com/gnormal/postgres-go) repo 包含了 [gnorm](https://github.com/gnormal/gnorm) 使用的模板，并做了一些假设，让我回去修改我的数据库模式。来自[自述](https://github.com/gnormal/postgres-go/blob/master/README.md):

> 这些模板假设了一些事情——如果一个列有一个默认的主键，我们将在插入时忽略这个主键，让数据库生成这个键(例如，一个 uuid 或自动递增的整数)。这避免了 Go 代码的零值是有效值的问题，因此您忘记了设置 id，它被插入 ID == 0(这通常是错误的)。
> 
> 其他假设—名为 updated_at 和 created_at 的列被假设为由数据库生成，因此永远不会手动插入或更新，并且在插入时使用 Postgres 的 RETURNING 语句返回(连同 id)。

下面是我完全修改过的数据库模式:

```
CREATE OR REPLACE FUNCTION trigger_set_timestamp ()
  RETURNS TRIGGER
  AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;DROP TABLE IF EXISTS streaming_plays;
DROP TABLE IF EXISTS tracks;
DROP TABLE IF EXISTS artists;CREATE TABLE artists (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);CREATE TABLE tracks (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  artist_id INTEGER NOT NULL REFERENCES artists (id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);CREATE TABLE streaming_plays (
  id SERIAL PRIMARY KEY,
  end_time TIMESTAMPTZ NOT NULL,
  ms_played INTEGER NOT NULL,
  track_id INTEGER NOT NULL REFERENCES tracks (id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);CREATE TRIGGER set_timestamp
  BEFORE UPDATE ON artists
  FOR EACH ROW
  EXECUTE PROCEDURE trigger_set_timestamp ();CREATE TRIGGER set_timestamp
  BEFORE UPDATE ON tracks
  FOR EACH ROW
  EXECUTE PROCEDURE trigger_set_timestamp ();CREATE TRIGGER set_timestamp
  BEFORE UPDATE ON streaming_plays
  FOR EACH ROW
  EXECUTE PROCEDURE trigger_set_timestamp ();
```

有了这个加载到我的数据库中，我就能够用`gnorm gen`生成支持代码，并扩展我的 Go 程序来循环遍历我的播放历史，并为列出的艺术家、曲目和播放创建条目。[下面是我最后一个`main.go`的要点](https://gist.github.com/timburks/5a13e4cceb421220b96bd417b1d305bf)。

# 用 SQL 分析我的游戏历史

将我的历史加载到 PostgreSQL 后，生成我的“包装”列表只需要运行正确的 SQL 命令。这里有一个返回我播放时间最长的十首歌曲:

```
SELECT sum(ms_played)/(60*60*1000.0), artists.name, tracks.name
FROM streaming_plays,tracks,artists
WHERE track_id = tracks.id
AND artist_id = artists.id
GROUP BY tracks.name, artists.name
ORDER BY sum(ms_played) DESC LIMIT 10
```

用`psql -U postgres -d playtime -a -f toptracks.sql` 运行它产生了下表:

```
?column?      |           name           |         name          
--------------------+--------------------------+--------------------
 4.6896372222222222 | Rage Against The Machine | Killing In The Name
 3.6687563888888889 | Rage Against The Machine | Wake Up
 3.5106108333333333 | Dave Grohl               | Play
 3.4357252777777778 | Rage Against The Machine | Renegades Of Funk
 3.0494194444444444 | Rage Against The Machine | Know Your Enemy
 2.8526719444444444 | Rage Against The Machine | Bombtrack
 2.5247416666666667 | Rage Against The Machine | Bulls On Parade
 2.4658491666666667 | Gwen Stefani             | Harajuku Girls
 2.4145450000000000 | Rage Against The Machine | Take The Power Back
 2.3744705555555556 | Jessie Ware              | Alone
```

(如前所述，今年我和 RATM 一起跑了很多次！)下面是一个查询，它返回了我的前十位艺术家:

```
SELECT sum(ms_played)/(60*60*1000), artists.name
FROM streaming_plays,tracks,artists
WHERE track_id = tracks.id
AND tracks.artist_id = artists.id
GROUP BY artists.name
ORDER BY sum(ms_played) DESC LIMIT 10?column? |           name           
----------+--------------------------
       54 | Rage Against The Machine
       31 | AC/DC
       18 | Foo Fighters
       15 | Hole
       15 | Britney Spears
       13 | Cibo Matto
       13 | Nine Inch Nails
       12 | Beastie Boys
       11 | Rush
       10 | Weezer
```

我还没有进一步探索，但可以想象按月、按一周中的某一天、或按一天中的某个时间来排列我的热门曲目(“热门锻炼曲目”、“热门早间音乐”等)。

# 使用 Hasura 探索我与 GraphQL 的历史

2018 年宣布， [Hasura](https://hasura.io/) 为 PostgreSQL 数据库创建 GraphQL 接口。稍微阅读了一下，我发现我可以在 Docker 容器中运行 Hasura，并将其附加到我的本地数据库:

```
#!/bin/bash
docker run -d --net=host \
       -e HASURA_GRAPHQL_DATABASE_URL=postgres://postgres:test@localhost:5432/playtime \
       -e HASURA_GRAPHQL_ENABLE_CONSOLE=true \
       hasura/graphql-engine:v1.0.0-rc.1
```

这给了我一个本地控制台，我可以用它来研究我的数据库，只需几次点击，我就可以接受 Hakura 的缺省值，并将我的键关系转换成一个丰富的 GraphQL 模式。这是一个 GraphQL 查询，查询的是歌手的歌曲:

![](img/e9cfbb37cf3662d7a2ece8ed8f41dd31.png)

这里有一个查询，显示每次我流[纸浆](https://open.spotify.com/artist/36E7oYfz3LLRto6l2WmDcD?si=2DyZcn8eSN2CUuzUyXHYvw)的[普通人](https://open.spotify.com/track/2fXKyAyPrEa24c6PJyqznF?si=pJa3ZqCGT3eZrw_mZEeVLQ):

![](img/11f7d528ed159d0ca194fce57e7c6731.png)

# 从 Spotify API 获取摘要和更多内容

有了 Spotify APIs，可以实时获得更多信息。这里有一个很好的教程，描述了如何使用 React 和 Node 查看您的实时 Spotify 历史。它使用 [Spotify Web API](https://developer.spotify.com/documentation/web-api/) 来[获取当前用户最近播放的曲目](https://developer.spotify.com/documentation/web-api/reference/player/get-recently-played/)。其他 API 函数包括 one to [获取用户的顶级艺术家和曲目](https://developer.spotify.com/documentation/web-api/reference/personalization/get-users-top-artists-and-tracks/)和方法[获取关于用户当前回放的信息](https://developer.spotify.com/documentation/web-api/reference/player/get-information-about-the-users-current-playback/)和[开始/恢复用户的回放](https://developer.spotify.com/documentation/web-api/reference/player/start-a-users-playback/)。有人想建一个[共享点唱机](https://en.wikipedia.org/wiki/Turntable.fm)吗？