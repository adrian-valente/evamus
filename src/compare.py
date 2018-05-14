from utils.comparison import comparison, analyse_and_compare

# comparison("/Users/Flo/Documents/PhD/Code/nipsmusic/data/JSB_Chorales/midi/", 
# 	["/Users/Flo/Documents/PhD/Code/nipsmusic/save/MLP/JSB_Chorales/midi/",
# 	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/InterpretableLossBachProp/JSB_Chorales/midi/",
# 	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/polyphonicDAC/JSB_Chorales/midi/",
# 	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/RandMBachProp/JSB_Chorales/midi/"], 
# 	['MLP',
# 	'InterpretableLossBachProp',
# 	'PolyphonicDAC',
# 	'RandMBachProp'], 
# 	write_report=True)


comparison("/Users/Flo/Documents/PhD/Code/nipsmusic/data/BachMusic21/midi/", 
	["/Users/Flo/Documents/PhD/Code/nipsmusic/save/InterpretableLossBachProp/BachMusic21/midi/",
	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/polyphonicDAC/BachMusic21/midi/",
	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/indepBachProp/BachMusic21/midi/",
	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/BachProp/BachMusic21/midi/",
	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/DeepBach/ChoralesMusic21/midi/",
	"/Users/Flo/Documents/PhD/Code/nipsmusic/save/MLP/BachMusic21/midi/"], 
	['interpLoss',
	'DAC',
	'Indep',
	'BachProp',
	'DeepBach',
	'MLP'], 
	write_report=True)