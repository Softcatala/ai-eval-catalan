<!-- alignment: center -->
<!-- no_footer -->
<!-- jump_to_middle -->

<span style="color: #b4ccff">**Avaluació d'IA, context i marc de treball**</span>

<span style="color: #b6eada">**Jordi Mas · Softcatalà**</span>  
<span style="color: #b6eada">GitHub: jordimas · X: @jordimash</span>

<!-- end_slide -->

# Objectiu

El nostre objectiu és contestar la pregunta: quina IA recomanem als usuaris domèstics?

Per respondre aquesta pregunta, hem de tenir en compte:

- Maquinari limitat (diferents recomanacions per diferent maquinari)

- Reproduïm les condicions reals d'ús.
    - Per exemple: llama.cpp com a motor d'inferència, quantificació, etc.

- En el cas dels LLM, característiques que són rellevants
    - Tasques: traducció, reformulació, etc.
    - Comportament: seguiment d'instruccions, mantenir la llengua, etc.

<!-- end_slide -->

# Apostem per dos tipus d'avaluacions

- Automàtiques: corpus i mètriques
    - Iniciativa: [Models en català](https://www.softcatala.org/ia-local/models-en-catala/)
    - Limitacions: les mètriques no sempre reflecteixen les preferències humanes, contaminació, errades en corpus / harness, etc.
- Humanes: avaluacions humanes
    - Iniciativa: [Arena.cat](https://github.com/Softcatala/arena-cat)
    - Limitacions: discrepància entre humans, capacitat d'avaluar allò que se't demana, etc.
    - Repte: captar i mantenir prou avaluadors (objectiu inicial: uns 20, amb unes 3 h per persona).

<!-- end_slide -->
<!-- no_footer -->

# Arena Cat

<span style="color: #ef5350">**Avaluació humana de models d'IA en català**</span>

**Reformulació** · 25 de 90 avaluacions · 65 pendents

<span style="color: #ef5350">━━━━━━</span><span style="color: #8892a6">━━━━━━━━━━━━━━━━</span>

<span style="color: #a5d7e8">**INDICACIÓ**</span> Reformula el text perquè el comprengui un infant de 10 anys.

**Text original (fragment):** La derivada d'una funció en un punt es defineix com el límit del quocient incremental i quantifica la taxa de variació instantània.

<!-- column_layout: [1, 1] -->
<!-- column: 0 -->

<span style="color: #a5d7e8">**RESPOSTA A · fragment**</span>

Imagina que la derivada és com un velocímetre: et diu exactament quina velocitat tens en aquest mateix instant.

<!-- column: 1 -->

<span style="color: #b6eada">**RESPOSTA B · fragment**</span>

La derivada d'una funció en un punt ens diu com canvia la funció en aquell moment.

<!-- reset_layout -->
<!-- new_line -->
<!-- alignment: center -->

<span style="color: #ef5350">**[ A és millor ]   [ B és millor ]   [ Empat ]   [ Cap de les dues ]**</span>

<span style="color: #8892a6">Recreació de la interfície · textos abreujats</span>

<!-- end_slide -->

# Què volem aportar a l'ecosistema?

- **Conjunts de dades reutilitzables**
    - Preferències humanes a partir dels resultats d'Arena.
    - Traducció i correcció basades en patrons reals de producció.
- **Avaluacions obertes i rellevants per a l'usuari final**
    - Benchmarks com mantic (Catalan drift).
    - Eines i metodologia per reproduir les avaluacions.
- **Resultats útils per a la comunitat**
    - Identificació de mancances per orientar la millora dels models.
    - Recomanacions per als usuaris domèstics segons la tasca i el maquinari.

Volem compartir resultats i contrastar criteris amb AINA/BSC i la resta de la comunitat.
