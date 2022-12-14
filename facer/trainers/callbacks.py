from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   patience=3,
   verbose=False,
   mode='min'
)

MODEL_CKPT_PATH = '../model/'
MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.5f}'

checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath=MODEL_CKPT_PATH,
                                      filename=MODEL_CKPT,
                                      save_top_k=3,
                                      mode='min')




