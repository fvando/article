from scipy.ndimage import zoom

# Proporcionalmente ajustar os arrays para 576, 480 e 96 slots
slots_576 = zoom(adjusted_needs, 576 / len(adjusted_needs), order=1).astype(int)
slots_480 = zoom(adjusted_needs, 480 / len(adjusted_needs), order=1).astype(int)
slots_96 = zoom(adjusted_needs, 96 / len(adjusted_needs), order=1).astype(int)

# Garantir que a soma seja ajustada proporcionalmente para manter consistência com o total original
slots_576 = (slots_576 * sum(adjusted_needs) // sum(slots_576)).tolist()
slots_480 = (slots_480 * sum(adjusted_needs) // sum(slots_480)).tolist()
slots_96 = (slots_96 * sum(adjusted_needs) // sum(slots_96)).tolist()

slots_576, slots_480, slots_96
