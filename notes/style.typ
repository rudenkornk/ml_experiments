#let ink = rgb("193747")
#let muted = rgb("52636b")
#let E = math.op("E")
#let Cov = math.op("Cov")
#let Var = math.op("Var")
#let tr = math.op("tr")
#let rank = math.op("rank")
#let diag = math.op("diag")
#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)
#let sign = math.op("sign")
#let prox = math.op("prox")

#let notes(body) = {
  set document(title: "Машинное обучение — плотный конспект четырёх занятий", author: "По занятиям Никиты Волкова")
  set page(
    paper: "a4",
    columns: 2,
    margin: (x: 12mm, top: 14mm, bottom: 13mm),
    header: text(font: "PT Sans", size: 8pt, fill: muted)[
      Машинное обучение · Никита Волков #h(1fr) Сентябрь–октябрь 2026
    ],
    footer: context align(center, text(font: "PT Sans", size: 8pt, fill: muted)[
      #counter(page).display("1") / #counter(page).final().first()
    ]),
  )
  set text(font: "PT Serif", size: 10.3pt, lang: "ru", hyphenate: true)
  set par(justify: true, leading: 0.43em, spacing: 4.5pt)
  set heading(numbering: none)
  show heading.where(level: 1): set text(font: "PT Sans", size: 14pt, weight: "bold", fill: ink)
  show heading.where(level: 2): set text(font: "PT Sans", size: 11pt, weight: "bold", fill: ink)
  show heading.where(level: 2): set block(above: 7pt, below: 3.5pt)
  show math.equation: set text(font: "STIX Two Math")
  show math.equation.where(block: true): set block(above: 4pt, below: 4pt)
  set list(indent: 9pt, body-indent: 4pt, spacing: 2.5pt)
  set enum(indent: 10pt, body-indent: 4pt, spacing: 2.5pt)
  set table(inset: 4pt, stroke: 0.35pt + rgb("b7c4ca"))
  show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
  body
}

#let chapter(title, subtitle, body) = {
  heading(level: 1, title)
  text(font: "PT Sans", size: 8.3pt, fill: muted, subtitle)
  v(2mm)
  body
}

#let sheet(title, subtitle, left, right) = {
  block(above: 6pt, below: 4pt)[
    #text(font: "PT Sans", size: 8.3pt, fill: muted)[*#title.slice(5)* · #subtitle]
  ]
  left
  right
}

#let key(body) = block(
  width: 100%, inset: 6pt, fill: rgb("eef4f6"), stroke: (left: 1.5pt + ink),
  above: 5pt, below: 5pt, body,
)

#let fig(path, source, caption) = block(breakable: false, above: 5pt, below: 5pt)[
  #image(path, width: 100%)
  #v(2pt)
  #text(font: "PT Sans", size: 7.8pt, fill: muted)[#caption · #source]
]
