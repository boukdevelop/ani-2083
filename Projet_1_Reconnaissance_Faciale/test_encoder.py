from face_encoder import load_encodings
enc, names, meta = load_encodings("test_encoded.pkl")
print(len(enc), len(names), meta["n_valid"])