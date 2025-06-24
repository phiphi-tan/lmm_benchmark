from .benchmark_tools import eval_results
from word2number import w2n


ref_list =   ['Performance', 'Sport', '...period.', 'GTOR®', '12', 'CAOL', 'DISTILLERY', 'G-ATCO', 'OUR', 'NEIGHBORS,', 'THE', 'FRIENDS', 'Muhomah', 'friends', '97215', 'NORTHWEST', 'THE', 'WHITEHEADED', 'ÚJSÁG', "WAGNER'ÚR?", 'PARTI', 'Chrstphre', 'Campbell', '903', 'West', 'Spofford', 'Unit', 'Seven', 'Spokane', 'Washington', '99205', '·DEPART', 'PARTI', 'DAN', 'DES', 'USE', 'KA', '003-', 'OV', 'FORTY', 'NINERS', 'S', 'conesa', 'conesa', 'Fanta', 'pper.', 'Pepper.', 'Imperial', 'Dr', 'Pepper', 'Dr', 'Pep', 'Pe', 'D', "mickey's", 'mainsall', 'Mainsall', 'WALT', 'DISNEY', 'STUDIOS', 'PgDn', 'PgUp', 'Home', 'Microsoft']
pred_list =  [' Performance.', ' Sport.', ' Period.', ' GTob.', ' Blue.', ' Caol.', ' Paul Tea Distillery.', ' G-ATCO.', ' Our.', ' NEIGHBORS.', ' THE.', ' Friends.', ' Muham.', ' Friends.', ' 97215.', ' Northwest.', ' The.', ' Whiteheaded.', ' Usag,', ' USAG WAGNER?', ' Online Bottleville, Stix 66600.', ' Christophe.', " Campbell's.", ' 903.', ' West.', ' Spofford.', ' UNIT.', ' Seven.', ' Spokane.', ' Washington.', ' 99205', ' Departure.', ' Bell.', ' The image is a close-up of a book cover. The book cover is white with a grayish-blue title. The title is "DANGEROUS" and is in a large, bold font. The background of the book cover', ' DES.', ' Use.', ' KA.', ' 033.', ' OV.', ' Forty.', ' NIGGERS.', ' 69.', ' Conesa.', ' Conea.', ' Fanta.', ' Imperial Soda.', ' Dr. Pepper.', ' Imperial Tiger.', ' Dr.', ' Dr.', ' Pepsi.', ' DR PEP.', ' Red.', ' Red.', " Mickey's.", ' Mainsaw.', ' New York City.', ' WALT.', ' Disney.', ' Studios.', ' Pgdn.', ' Pgup.', ' Home.', ' Microsoft.']
# pred_list = [str(w2n.word_to_num(i)) for i in pred_list]
pred_list = [i.strip()[:-1] for i in pred_list]
print(pred_list)

print(eval_results(ref_list, pred_list, 'exact_match'))