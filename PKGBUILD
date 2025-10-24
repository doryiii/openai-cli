# Maintainer: Dory <dory@dory.moe>
pkgname=openai-cli-git
_pkgname=openai-cli
pkgver=r5.e99f3d4
pkgrel=1
pkgdesc="A simple and feature-rich CLI for OpenAI's chat API, with support for tools and image inputs."
arch=('any')
url="https://github.com/doryiii/openai-cli"
license=('MIT')
depends=('python' 'python-requests' 'python-termcolor' 'python-rich' 'python-html2text')
makedepends=('git')
source=("git+https://github.com/doryiii/openai-cli.git")
sha256sums=('SKIP')

pkgver() {
  cd "$_pkgname"
  printf "r%s.%s" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

package() {
	cd "${srcdir}/${_pkgname}"
	install -Dm755 openai_chat.py "${pkgdir}/usr/bin/openai-cli"
}

