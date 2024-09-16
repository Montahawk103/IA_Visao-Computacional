Este projeto implementa um sistema de visão computacional para contar hambúrgueres e cestas em uma linha de produção, utilizando processamento de vídeo em tempo real.
Descrição

O Burger Counter analisa um vídeo de uma linha de produção de hambúrgueres e realiza as seguintes tarefas:

Conta o número de hambúrgueres que passam pela esteira.
Conta o número de cestas vazias que entram na linha.
Conta o número de cestas cheias que saem da linha.
Calcula o tempo médio de preenchimento de uma cesta.
Calcula o tempo médio que cada hambúrguer leva para passar pela esteira.

Requisitos

Python 3.7+
OpenCV
NumPy

Instalação

Clone este repositório:
Copygit clone https://github.com/Montahawk103/IA_Visao-Computacional.git
cd IA_Visao-Computacional

Instale as dependências:
Copypip install opencv-python numpy


Uso

Coloque o vídeo a ser analisado (nomeado como burger-trim.mp4) no mesmo diretório do script.
Execute o script:
Copypython main.py

O programa irá processar o vídeo e exibir uma visualização em tempo real. Pressione 'q' para encerrar o processamento.
Após o processamento, os resultados serão exibidos no console.

Estrutura do Projeto

main.py: O script principal contendo a classe BurgerCounter e a lógica de processamento.
burger-trim.mp4: O vídeo de entrada (não incluído no repositório).
requirements.txt: Bibliotecas necessárias.
pyenv: Utilização e MV.

Customização
Você pode ajustar vários parâmetros no início da classe BurgerCounter para otimizar a detecção:

min_basket_area e max_basket_area: Ajustam o tamanho das cestas detectadas.
min_burger_area e max_burger_area: Ajustam o tamanho dos hambúrgueres detectados.
Regiões de Interesse (ROIs): Podem ser ajustadas alterando os valores nas linhas que definem basket_entry_roi, basket_exit_roi, e burger_roi.

Contribuindo
Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter um Pull Request.
