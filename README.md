# KKB TEAM Kaggleリポジトリ

## バージョン管理規則

### pushやmergeについて

このリポジトリは，メンバーがありとあらゆるところに`push`できるようになっていますが，それをやるととても大変なことになるので，**必ず適切なbranchを用意してそこからpullrequestを出してチームのメンバーに見てもらってからマージする**ようにしてください．ここでいうチームとは例えばモデル班，前処理班などのことも指します．

### branch

`origin master`は本提出用のbranch，`origin develop`は開発branchです．`origin develop`に基本的にみんなのコードをmergeしていく形にして，みんなで話し合って最終提出するコードをmasterにmergeしていく形にします．なので基本的に`origin develop`からbranchを切ってください．あと`origin develop`含め，みんなが使いそうなbranchは**原則動くコード**がある状態にしてください．

### 作業例

モデル作成チームがモデルの改良をすることを考える．

※初めてdevelopブランチに入る時，

```bash
git checkout -b develop origin/develop
```

これでdevelopブランチは，**originの**developブランチを示すことになる．これやんないとコミットログいかれる．

developブランチからブランチを切る．どこからブランチ切るかは割と重要．デフォルトではcheckoutすると，直前にいたブランチから切られるはず(要検証)．[参考](https://www.granfairs.com/blog/staff/git-mistake-parent-branch)

```bash
git checkout develop
git checkout -b model
git checkout -b fix_pool
```

こうすればdevelopから切れてmodelにいって，modelから切れてfix_poolに飛べるのかな？

ここで作業する．作業後，

```bash
git add (file)
git commit -m "message"
git push (remote名) fix_pool
```

この後，このfix_poolをmodelブランチにプルリク出す．modelチームメンバーと相談して適切な変更だと思ったらmodelにマージ．そのあとdevelopにプルリクを出してみんなで相談して適切な変更だと思ったらdevelopにマージ．  

マージに関してはコンフリクト等の問題があるので柔軟にやる．

## Google Colab環境構築  

1. [このブログ](https://qiita.com/tamitarai/items/1c9da94fdfad997c3336)を参考に，各自のdriveにリポジトリをcloneする．ここでcloneする先は`"gdrive/My Drive/KKB-kaggle/"`とすること！！！`"gdrive/My Drive/"`と入力してしまうとあなたのGoogle Driveのファイル全てがgit追跡の対象になります．

1. `/KKB-kaggle/terminal.ipynb`というnotebookを作成し，そこでターミナル操作を行うようにする (ローカルでjupyter notebookを用いて`terminal.ipynb`というnotebookを作成し，`/KKB-kaggle/`にアップロードするのがいいかも？)．`terminal.ipynb`はgitに追跡されないように`.gitignore`に登録されています．

1. 各コンペディレクトリのREADMEを参照し，指定の操作を行う．

## ローカル環境構築(WIPです．無視してください)

**ローカル環境の構築は現在推奨していません．WIPです． 以下の方法ではなく， Colabで直接編集する方法を採用してください．**

基本的にGoogle Colabのバージョンと完全一致させます．自分の環境ではうまくいったけどXXさんの環境ではうまくいかなかったということがないように，統一をよろしくお願いします．特に深層学習のフレームワークではそういうことが起こりがちです．

### Pythonバージョン

Python 3.6.9  

※Python環境の構築については，下の方に書いてあります．

### 必要なpipモジュールのインストール方法

```bash
pip3 install -r requirements.txt
```

何度もいいますが**モジュールのバージョンの統一**をよろしくお願いします．

### 指定したPythonバージョンの設定方法

#### MacOSの場合

HomeBrewでは旧バージョンのPythonがインストールできないのでpyenv virtualenvを使って環境構築をします．Anacondaでも環境分けできるならそれでもいいですが，ここには書きません．一番避けるべきは**condaでインストールしたものとpipでインストールしたものが同一環境内に混在すること**です．  

まずはpyenvとpyenv-virtualenvをインストールします．

```bash
brew update
brew install pyenv
brew install pyenv-virtualenv
```

Python3.6.9の環境をインストールします

```bash
pyenv install 3.6.9
```

もしこれに失敗するようだったら

```bash
brew update pyenv
```

を実行しましょう．

そしたら

```bash
cd ~
```

とホームディレクトリに移動し，pyenv.bashというファイルを作成してそこに，

```bash
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
```

と書き込みます．そして

```bash
source ~/pyenv.bash
```

とすると，pyenvでインストールしたpython3.6.9環境が使えるようになります．これはターミナルを立ち上げるたびに実行しましょう．.bash_profile等に書いてもいいですが，個人的にはシステムのPython環境と分けたいのでいつもこうしてます．

これを実行する前とあとで`which python3`の実行結果が変わっていることから，違うPython環境が利用されていることがわかります．  

そうしたら

```bash
pyenv virtualenv 3.6.9 KKB-kaggle
activate KKB-kaggle
```

を実行しましょう．そうすることでKKB-kaggle用の環境に入ります．ここで

```bash
pip3 install -r requirements.txt
```

を入力すると，必要なモジュールが全て入ります．

