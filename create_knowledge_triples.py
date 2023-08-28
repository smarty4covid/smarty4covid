from smarty import get_p_q_dataframes
# from utils import get_audio
from owlready2 import *
import pandas as pd
import pickle
from tqdm import tqdm
import math
# import librosa

## Where is the ontology located
onto_path = 'smarty-ontology.owl'

## Where is the combined csv?
combined = 'combined_data.csv'
lab = 'labelstudio-03-03.csv'

lab_df = pd.read_csv(lab)
noise = dict()
valid = dict()
for i in range(lab_df.shape[0]):
	aud = '/'.join(lab_df['audio'][i].split('/')[-2:])
	noise[aud] = lab_df['background_noise'][i]
	valid[aud]=lab_df['valid_audio'][i]


num_experts=5

breath_characterizations=dict()
breath_characterizations['Audible_choking']='AudibleChoking'
breath_characterizations['Audible_nasal_congestion']='AudibleNasalCongestion'
breath_characterizations['Dyspnea_Shortness_of_breath']='AudibleDyspnoea'
breath_characterizations['Expiratory_stridor']='AudibleExpiratoryStridor'
breath_characterizations['Inspiratory_stridor']='AudibleInspiratoryStridor'
breath_characterizations['Other_and_unspecified_abnormalities_of_breathing']='AudibleAbnormality'
breath_characterizations['Prolonged_expiration']='AudibleProlongedExpiration'
breath_characterizations['Respiratory_crackles']='AudibleRespiratoryCrackles'
breath_characterizations['Stridor']='AudibleStridor'
breath_characterizations['Wheezing']='AudibleWheezing'
breath_characterizations['breath_depth']={'Cannot breathe deeply enough':'InsufficientBreathDepth'}

cough_characterizations=dict()
cough_characterizations['Audible_choking']='AudibleChoking'
cough_characterizations['Audible_dyspnea']='AudibleDyspnoea'
cough_characterizations['Audible_nasal_congestion']='AudibleNasalCongestion'
cough_characterizations['Audible_stridor']='AudibleStridor'
cough_characterizations['Audible_wheezing']='AudibleWheezing'
cough_characterizations['Barking_cough']='BarkingCoughCharacterization'
cough_characterizations['Croupy_cough']='CroupyCoughCharacterization'
cough_characterizations['Dry']='DryCoughCharacterization'
cough_characterizations['Hacking_cough']='HackingCoughCharacterization'
cough_characterizations['Productive']='ProductiveCoughCharacterization'
cough_characterizations['cough_is']=dict()
cough_characterizations['cough_is']['Mild (from a sick person)']='MildCoughCharacterization'
cough_characterizations['cough_is']['Severe (from a sick person)']='SevereCoughCharacterization'
cough_characterizations['cough_is']['Pseudocough/Healthy cough (from a healthy person)']='PseudocoughCharacterization'
cough_characterizations['patient_has']=dict()
cough_characterizations['patient_has']['An upper respiratory tract infection']='AudibleUpperRespiratoryTractInfection'
cough_characterizations['patient_has']['Obstructive lung disease (Asthma, COPD, ...)']='AudibleObstructiveLungDisease'
cough_characterizations['patient_has']['A lower respiratory tract infection']='AudibleLowerRespiratoryTractInfection'


expert_inds=dict()
df = pd.read_csv(combined)
onto = get_ontology(onto_path).load()
with onto:
	for i in range(num_experts+1):
		expert_inds[i]=onto['Expert']()
	## Patients/users
	print('Patients...')
	for i in tqdm(range(df.shape[0])):
		pid = df['ParticipantId'][i]
		# print(pid)
		u = onto['User'](pid)
		if df['Sex'][i]==0:
			u.is_a.append(onto['Male'])
		elif df['Sex'][i]==1:
			u.is_a.append(onto['Female'])
		# Assert age
		age_categories = {0:'UserTwenties',1:'UserThirties',2:'UserFourties',3:'UserFifties',4:'UserSixties',5:'UserSeventies',6:'UserEighties'}
		u.is_a.append(onto[age_categories[df['AgeCategory'][i]]])
		
		## Assert height
		# u.hasHeight.append(df['Height'][i])
		
		## Assert Weight
		# u.hasWeight.append(df['Weight'][i])
		u.hasBMI.append(float(df['BMI'][i]))

		preconds = ['RespiratoryDeficiency','CysticFibrosis','PneumOther','CoronaryDisease','Hypertension','ValveDisease','HeartAttack','Stroke','CardiovascularOther','Diabetes','Kidney','Transplant','Cancer','HIV']
		u_preconds = [a for a in preconds if df[a][i]]
		if len(u_preconds)>0:
			for pc in u_preconds:
				pc_ind = onto[pc]()
				u.hasPreexistingCondition.append(pc_ind)


		# User Instance/submission
		ui = onto['UserInstance'](df['SubmissionId'][i])

		if df['Tested'][i] in ['positive','negative']:
			t = onto['CovidTest']()
			if df['Tested'][i]=='positive':
				t.is_a.append(onto['PositiveTest'])
			elif df['Tested'][i]=='negative':
				t.is_a.append(onto['NegativeTest'])
			if df['TestPCR'][i]:
				t.is_a.append(onto['PCRTest'])
			elif df['TestSelf'][i]:
				t.is_a.append(onto['SelfTest'])
			elif df['TestRapid'][i]:
				t.is_a.append(onto['RapidTest'])
			ui.hasCovidTest.append(t)



		## Hospitalised
		if df['Hospitalised'][i]=='1':
			ui.is_a.append(onto['HospitalizedUserInstance'])
		elif df['Hospitalised'][i]=='2':
			ui.is_a.append(onto['HospitalizedAWeekAgoUserInstance'])
		elif df['Hospitalised'][i]=='3':
			ui.is_a.append(onto['HospitalizedMonthsAgoUserInstance'])


		## Vaccinated
		if df['Vaccinated'][i] == 'no':
			ui.is_a.append(onto['Unvaccinated'])
		elif df['Vaccinated'][i]=='partially':
			ui.is_a.append(onto['PartiallyVaccinated'])
		elif df['Vaccinated'][i]=='fully':
			ui.is_a.append(onto['FullyVaccinated'])
		elif df['Vaccinated'][i]=='booster1':
			ui.is_a.append(onto['BoosterVaccinated'])
		
		## Exposed
		if df['Exposed'][i]=='yes':
			ui.is_a.append(onto['ExposedUserInstance'])
		elif df['Exposed'][i]=='maybe':
			ui.is_a.append(onto['MaybeExposedUserInstance'])
		elif df['Exposed'][i]=='no':
			ui.is_a.append(onto['NotExposedUserInstance'])


		## Travelled
		if not math.isnan(df['TravelledAbroad'][i]) and int(df['TravelledAbroad'][i])==0:
			ui.is_a.append(onto['NotTravelledAbroadUserInstance'])
		elif not math.isnan(df['TravelledAbroad'][i]) and int(df['TravelledAbroad'][i])==1:
			ui.is_a.append(onto['TravelledAbroadUserInstance'])
		

		onto['User'](df['ParticipantId'][i]).hasUserInstance.append(ui)


		## Symptoms
		symps = ['SoarThroat','DryCough','WetCough','Spit','RunnyNose','BreathDiscomfort','HasFever','Tremble','Fatigue','Headache','Dizziness','MuscleAche','TasteLoss','DiarrheaUpsetstomach','Sneezing','DryThroat']
		for s in symps:
			if str(df[s][i]) not in ['True','False']:
				df[s][i] = False

		u_symps = [s.replace('SoarThroat','SoreThroat') for s in symps if df[s][i]]
		if len(u_symps)>0:
			
			for s in u_symps:
				s_ind = onto['Symptom']()
				s_ind.is_a.append(onto[s])
				ui.hasSymptom.append(s_ind)
		
		## OxygenSat
		if df['OxygenSaturation'][i] is not None:
			ui.hasOxygenSaturation.append(float(df['OxygenSaturation'][i]))

		## BPM
		if df['BPM'][i] is not None:
			ui.hasBPM.append(float(df['BPM'][i]))

		## SystolicPressure
		if df['SystolicPressure'][i] is not None:
			ui.hasSystolicPressure.append(float(df['SystolicPressure'][i]))

		## Diastolic Pressure
		if df['DiastolicPressure'][i] is not None:
			ui.hasDiastolicPressure.append(float(df['DiastolicPressure'][i]))

		## Difficulties
		diffs = ['LeaveBed','LeaveHome','PrepareMeal','Concentrate','SelfCare','OtherDifficulty']
		u_diffs = [d for d in diffs if df[d][i]]
		if len(u_diffs)>0:
			d_ind = onto['Difficulty']()
			for d in u_diffs:
				if d!='OtherDifficulty':
					d_ind.is_a.append(onto[d+'Difficulty'])
				ui.hasDifficulty.append(d_ind)
		
		## Smoking
		if df['Smoking'][i]=='yes':
			ui.is_a.append(onto['SmokerInstance'])
			if df['YearSmoking'][i]!="No answer":
				if df['YearSmoking'][i] is not None and not math.isnan(df['YearSmoking'][i]):
					ui.hasYearsSmoking.append(int(df['YearSmoking'][i]))
			if df['Cigarettes'][i]!="No answer" and df['Cigarettes'][i]!='' and type(df['Cigarettes'])==str:
				ui.smokesCigarettesPerDay.append(int(df['Cigarettes'][i].replace('o','').replace('u','')))
		elif df['Smoking'][i]=='ex':
			ui.is_a.append(onto['PastSmokerInstance'])
			if df['YearsNonSmoker'][i]!='No answer' and df['YearsNonSmoker'][i]!='' and not math.isnan(df['YearsNonSmoker'][i]):
				ui.hasYearsNonSmoker.append(int(df['YearsNonSmoker'][i]))
		elif df['Smoking'][i]=='nev':
			ui.is_a.append(onto['NeverSmokedInstance'])

		## Vaping
		# print(df['Vaping'][i])
		if not math.isnan(df['Vaping'][i]) and int(df['Vaping'][i])==1:
			ui.is_a.append(onto['VaperInstance'])


		## Anxiety
		if not math.isnan(df['Covid19Anxiety'][i]):
			if int(df['Covid19Anxiety'][i])==0:
				ui.is_a.append(onto['NotAnxiousUserInstance'])
			elif int(df['Covid19Anxiety'][i])==1:
				ui.is_a.append(onto['ABitAnxiousUserInstance'])
			elif int(df['Covid19Anxiety'][i])==2:
				ui.is_a.append(onto['NotAnxiousUserInstance'])
			elif int(df['Covid19Anxiety'][i]) in [3,4]:
				ui.is_a.append(onto['VeryAnxiousUserInstance'])



		## Employment
		if df['Working'][i]=='store':
			ui.is_a.append(onto['WorkAtStoreUserInstance'])
		elif df['Working'][i]=='home':
			ui.is_a.append(onto['WorkFromHomeUserInstance'])
		elif df['Working'][i]=='hospital':
			ui.is_a.append(onto['WorkAtHospitalUserInstance'])
		elif df['Working'][i]=='social':
			ui.is_a.append(onto['WorkAtServiceUserInstance'])


		## Hold your breath
		ui.canHoldBreath.append(int(df['HoldYourBreath'][i]))

		cough_id = df['ParticipantId'][i]+'/questionnaire.'+df['SubmissionId'][i]+'.audio.cough.mp3'
		cough_path = cough_id#data_path+'/'+cough_id
		breath2_id = df['ParticipantId'][i]+'/questionnaire.'+df['SubmissionId'][i]+'.audio.breath_2.mp3'
		breath2_path = breath2_id#data_path+'/'+breath2_id
		speech_id = df['ParticipantId'][i]+'/questionnaire.'+df['SubmissionId'][i]+'.audio.speech.mp3'
		speech_path = speech_id

		# if cough_id in valid:
		if cough_id in valid and valid[cough_id]=='Yes':
			# print('is valid!')
			c_ind = onto['CoughAudio']()
			c_ind.hasFileName.append(cough_path)
			ui.hasCoughAudio.append(c_ind)
			if noise[cough_id]=="Poor":
				c_ind.is_a.append(onto['PoorQualityAudio'])
			elif noise[cough_id]=='OK':
				c_ind.is_a.append(onto['OKQualityAudio'])
			elif noise[cough_id]=='Good':
				c_ind.is_a.append(onto['GoodQualityAudio'])

			for e in [1,2,3,4,5]:
				for k in cough_characterizations:
					if df['expert_'+str(e)+'_cough_'+k][i] is not None:
						if type(cough_characterizations[k])==str:
							if df['expert_'+str(e)+'_cough_'+k][i]=='TRUE':
								char_ind=onto[cough_characterizations[k]]()
								char_ind.characterizedBy.append(expert_inds[e])
								c_ind.hasCharacterization.append(char_ind)
						elif df['expert_'+str(e)+'_cough_'+k][i] in cough_characterizations[k]:
							char_ind=onto[cough_characterizations[k][df['expert_'+str(e)+'_cough_'+k][i]]]()
							char_ind.characterizedBy.append(expert_inds[e])
							c_ind.hasCharacterization.append(char_ind)

			


		# if breath2_id in valid:
		if breath2_id in valid and valid[breath2_id]=='Yes':
			c_ind = onto['DeepBreathingAudio']()
			c_ind.hasFileName.append(breath2_path)
			ui.hasDeepBreathingAudio.append(c_ind)
			if noise[breath2_id]=="Poor":
				c_ind.is_a.append(onto['PoorQualityAudio'])
			elif noise[breath2_id]=='OK':
				c_ind.is_a.append(onto['OKQualityAudio'])
			elif noise[breath2_id]=='Good':
				c_ind.is_a.append(onto['GoodQualityAudio'])

			# Expert annotations
			for e in [1,3,4,5]:
				for k in breath_characterizations:
					if df['expert_'+str(e)+'_breath_'+k][i] is not None:
						if type(breath_characterizations[k])==str:
							if df['expert_'+str(e)+'_breath_'+k][i]=='TRUE':
								# print(breath_characterizations[k])
								char_ind = onto[breath_characterizations[k]]()
								char_ind.characterizedBy.append(expert_inds[e])
								c_ind.hasCharacterization.append(char_ind)
						elif k=='breath_depth' and df['expert_'+str(e)+'_breath_'+k][i]=='Cannot breathe deeply enough':
							char_ind=onto['InsufficientBreathDepth']()
							char_ind.characterizedBy.append(expert_inds[e])
							c_ind.hasCharacterization.append(char_ind)



		# if speech_id in valid:
		if speech_id in valid and valid[speech_id]=='Yes':
			c_ind = onto['SpeechAudio']()
			c_ind.hasFileName.append(speech_path)
			ui.hasSpeechAudio.append(c_ind)
			if noise[speech_id]=="Poor":
				c_ind.is_a.append(onto['PoorQualityAudio'])
			elif noise[speech_id]=='OK':
				c_ind.is_a.append(onto['OKQualityAudio'])
			elif noise[speech_id]=='Good':
				c_ind.is_a.append(onto['GoodQualityAudio'])

		
		if not df['NoBreath_1'][i]:
			b1_ind = onto['RegularBreathingAudio']()
			pth = df['ParticipantId'][i]+'/questionnaire.'+df['SubmissionId'][i]+'.audio.breath_1.mp3'
			b1_ind.hasFileName.append(pth)
			ui.hasRegularBreathingAudio.append(b1_ind)

		
	onto.save('smarty-triples.nt', format = "ntriples")





