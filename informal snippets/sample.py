import fxcmpy
con=fxcmpy.fxcmpy(config_file='fxcm.cfg')
print(con.is_connected())
