# pgmpy-modified
Engenharia reversa de redes de regulação gênica por meio de modelos gráficos probabilísticos

Este repositório possui uma extensão da biblioteca _pgmpy_ implementada em 
python. Pgmpy é uma biblioteca que implementa modelos gráficos probabilísticos, em especial o modelo de redes Bayesianas proposto por Koller & Friedman em _Probabilistic Graphical Models: principles and techniques, 2009_. 

Esta extensão foi utilizada a fim de realizar a inferência de redes de regulação gênica, utilizando como entrada dados temporais de expressão gênica. Para isto, foram realizadas algumas modificações que permitissem que o modelo lidasse com dados de séries temporais. Abaixo, são listadas estas modificações:
1. Foi proposta uma alteração no modelo de redes Bayesianas estático, levando em conta que em dados temporais de expressão gênica, um determinado gene altera seu comportamento no tempo _t_ em decorrência do comportamento de seus genes reguladores presentes no tempo _t-1_.
1. Implementação do modelo de redes Bayesianas dinâmicas, proposto por _Nir Friedman, Kevin Murphy e Stuart Russell em Learning the structure of dynamic probabilistic networks, 1998._
1. Permitir que o usuário adicione um arquivo contendo relações comprovadas entre genes, onde tais relações podem ser retiradas de bancos de dados biológicos. Cada relação deve conter uma pontuação, variando de 0 a 1, que indica o quanto os genes estão relacionados. Esta pontuação contribuirá no processo de inferência de redes, durante o cálculo da melhor estrutura de rede que representa o comportamento dos dados de entrada.

