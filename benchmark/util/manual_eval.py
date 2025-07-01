from .benchmark_tools import eval_results
from word2number import w2n
import ast


# ref_list =   ['Rhetorical Terms', 'Kaplan', 'The Fall', 'Stefano lacovella', 'Gary Dessler', 'The News', 'Clark Kidder', 'Angela Black', 'Rich Froning', 'Hiroshima', 'Sacred Knowledge', 'Shawn Rashid', 'Dick Couch', 'George E. Dieter, Linda C. Schmidt', 'Vanishing Gourds', 'Scot Ober', 'Anthony J. Martin', 'Ben Walker', 'Eckankar', 'Unknown', 'Pet Business Planning Almanac', 'Lorene Wales', 'Mason Douglas', 'The Wisdom Books', 'Vassili Zaitsev', 'Ectectus Parrots', 'Being digital', 'David Twicken', 'Dragons', 'Sarah Dessen', 'The Frozen Echo', 'GRE Guide 2015', 'Radium Girls', 'The Imitation of Christ', 'Jane Eyre', 'Multiple authors', 'Harald Stumpe', 'Joseph Correa', 'Charles Timmersman', 'Entrepreneurial Life Plan', 'Truman Smith', 'Terri Reed', 'Sawyer Bennett', 'Captain James Cook', 'Produded Hooking', 'Bruce Bliven, Jr.', 'Dick Weiss', 'Charles Devereux', 'SharePoint 2010 Business Connectivity Services', 'Sanjay Acharya', 'People', 'The Invention of World Religions', 'Michael Morpurgo', 'Mary G. Houston', 'Ken McLinton', 'Unknown', "Eiffel's Tower", 'Fluoride Toxicity in Animals', 'Math', 'Going Low', 'Thanks for the Feedback', 'Pete Whitehead', 'Trade Secrets', 'Anthony Cohen']
# pred_list =  [' A Handlist of Rhetorical Terms ', ' Kaplan ', ' The Fall: A Novel ', ' Stefano Iacovella ', ' Gary Dessler ', " The News: A User's Manual ", ' Clark Kidder ', ' Angela Black ', ' Rich Froning ', ' John Hersey: Hiroshima ', ' Sacred Knowledge: Psychedelics and Religious Experiences ', ' Shawn Rashid ', ' Sua Sponte ', ' George E. Dieter ', ' The Vanishing Gourds: A Sukkot Mystery ', ' Scot Ober ', ' Anthony J. Martin ', ' Ben Walker ', ' Eckankar ', ' MegaCalendars ', ' Pet Business Planning Almanack 2016 ', ' Lorene Wales ', ' Mason Douglas ', ' The Wisdom Books: Job, Proverbs, and Ecclesiastes ', ' Vassili Zaitsev ', " Eclectus Parrots: A Complete Pet Owner's Manual ", ' Being Digital ', ' David Twicken ', ' How to Train Your Dragon 2: Draw-It Dragons ', ' Sarah Dessen ', ' The Frozen Echo: Greenland and the Exploration of North America, ca. A.D. 1000-1500 ', " Gruber's Complete GRE Guide 2015: The Easiest, Fastest Way to Improve Your Score ", ' Radium Girls ', ' The Imitation of Christ ', " The image you've provided appears to be a word cloud, which is a visual representation of text data where the size of each word indicates its frequency or importance. The words in the image are arranged in a way that forms a cohesive phrase", ' Marianne Fleming ', ' Harald Stumpke ', ' Joseph Correa (Certified Sports Nutritionist) ', ' Charles Timmerman ', ' Less Work, More Money: The Entrepreneurial Life Plan ', ' Truman Smith ', ' Terri Reed ', ' Sawyer Bennett ', ' A. Grenfell Price ', ' Prodded Hooking for a Three-Dimensional Effect ', ' Bruce Bliven ', ' Jim Sumner ', ' Charles Devereux ', ' Microsoft SharePoint 2010 Business Connectivity Services ', ' Sanjay Acharya ', ' People Magazine ', ' The Invention of World Religions ', ' Michael Morpurgo ', ' Mary G. Houston ', ' Ken McClinton ', ' Susan C. Eaton ', " Eiffel's Tower: The Thrilling Story Behind Paris's Beloved Monument and the Extraordinary World's Fair That Introduced It ", ' Fluoride Toxicity in Animals (SpringerBriefs in Animal Sciences) ', ' DK Workbooks: Math, Grade 1 ', ' Going Low: How to Break Your Individual Golf Scoring Barrier by Thinking Like a Pro ', ' Thanks for the Feedback: The Science and Art of Receiving Feedback Well ', ' Highlights for Children ', ' Trade Secrets of a Haircolor Expert: How Haircolor Really Works ', ' Anthony Cohen ']
# # pred_list = [str(w2n.word_to_num(i)) for i in pred_list]
# pred_list = [i.strip() for i in pred_list]
# print(pred_list)

# print(eval_results(ref_list, pred_list, 'exact_match'))

# score_list = [0.0, 0.0, 0.61, 0.0, 0.24, 0.48, 0.03, 0.0, 0.7, 0.54, 0.82, 0.85, 0.81, 0.89, 0.51, 0.0, 0.0, 0.61, 0.31, 0.0, 0.63, 0.84, 0.67, 0.58, 0.0, 0.78, 0.06, 0.54, 0.67, 0.75, 0.51, 0.57, 0.55, 0.34, 0.88, 0.0, 0.15, 0.43, 0.54, 0.0, 0.81, 0.62, 0.1, 0.0, 0.82, 0.65, 0.74, 0.4, 0.0, 0.38, 0.75, 0.59, 0.49, 0.0, 0.0, 0.84, 0.0, 0.0, 0.0, 0.89, 0.78, 0.0, 0.75, 0.77] 
# avg = sum(score_list) / len(score_list)
# print(round(avg, 2))

pred = ['[62,291,170,378]', '[16,142,37,156]', '[378,196,1987,1254]', '[95,487,620,761]', '[38,62,190,157]', '[10,5,240,130]', '[72,49,218,99]', '[73,29,145,180]', '[38,56,432,307]', '[10,10,200,150]', '[139,156,2897,1380]', '[10,10]', '[28,23,301,274]', '[10,10], [200,150]', '[10,10]', '[514,60,839,247]', '[194,206,273,245]', '[10,10], [200,150]', '[73,40,859,592]', '[135,248,176,335]', '[235,41,1030,667]', '[10,10][200,150]', '[253,148,1740,946]', '[1,124,336,504]', '[90,302,145,387]', '[0,34,438,345]', '[45,60,132,107]', '[10,10]', '[0,123,584,504]', '[54,36,476,291]', '[10,10,200,150]', '[127,14,439,360]', '[30,85,486,329]', '[453,168,1057,552]', '[10,10][200,150]', '[327,410,420,465]', '[165,200,379,471]', '[30,92,468,261]', '[61,63,312,315]', '[287,79,473,265]', '[49,83,337,295]', '[10,10]', '[145,83,346,209]', '[137,42,230,98]', '[10,10,200,180]', '[10,10]', '[82,30,665,261]', '[90,148,643,427]', '[154,78,209,263]', '[10,10,120,120]', '[37,0,219,86]', '[10,10]', '[10,10][20,20]', '[268,3,437,501]', '[798,0,1235,924]', '[10,10][200,200]', '[36,67,102,125]', '[176,120,285,269]', '[1,204,178,413]', '[25,17,483,289]', '[10,10,200,150]', '[276,78,504,319]', '[94,0,504,364]', '[10,10][200,150]']

new_pred = []

for p in pred:
    try:
        new_pred.append(ast.literal_eval(p))
    except:
        new_pred.append(p)

print(pred)
print(new_pred)
