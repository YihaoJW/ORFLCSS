import tensorflow as tf


class RotaryEmbedding(tf.keras.layers.Layer):
    """Rotary positional encoding layer.

    This layer encodes absolute positional information with a rotation matrix.
    It calculates the rotary encoding with sine and cosine functions that use
    geometrically increasing wavelengths, as defined in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864v4).

    The input tensor must have a sequence and a feature dimension. It can be
    of shape `(batch_size, sequence_length, feature_length)` or with an extra
    multi-head dimension `(batch_size, sequence_length, num_heads, feature_length)`.

    Args:
        max_wavelength: int. Maximum angular wavelength of the sine/cosine curves.
        scaling_factor: float. Scaling factor used to scale token positions.
        sequence_axis: int. The axis representing the sequence dimension.
        feature_axis: int. The axis representing the feature dimension.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            max_wavelength=10000,
            scaling_factor=1.0,
            sequence_axis=1,
            feature_axis=-1,
            **kwargs
    ):
        super(RotaryEmbedding, self).__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.scaling_factor = scaling_factor
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis

    def call(self, inputs, start_index=0, positions=None):
        # Move the designated feature axis to the last position and sequence axis to position 1.
        inputs = self._moveaxis(inputs, (self.feature_axis, self.sequence_axis), (-1, 1))
        cos_emb, sin_emb = self._compute_cos_sin_embedding(inputs, start_index, positions)
        output = self._apply_rotary_pos_emb(inputs, cos_emb, sin_emb)
        # Move the axes back to their original positions.
        output = self._moveaxis(output, (-1, 1), (self.feature_axis, self.sequence_axis))
        return output

    @staticmethod
    def _moveaxis(x, source, destination):
        """Rearranges the axes of tensor x, similar to np.moveaxis."""
        rank = len(x.shape)
        # Normalize negative indices.
        source = [s if s >= 0 else s + rank for s in source]
        destination = [d if d >= 0 else d + rank for d in destination]
        # Build the new order.
        order = list(range(rank))
        for s, d in sorted(zip(source, destination), key=lambda pair: pair[1]):
            order.remove(s)
            order.insert(d, s)
        return tf.transpose(x, perm=order)

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        # Split the last dimension into two halves.
        x1, x2 = tf.split(tensor, num_or_size_splits=2, axis=-1)
        # Create the rotated tensor by stacking -x2 and x1.
        half_rot_tensor = tf.stack([-x2, x1], axis=-2)
        half_rot_tensor = tf.reshape(half_rot_tensor, tf.shape(tensor))
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _compute_positions(self, inputs, start_index=0):
        # Use the sequence length from the tensor shape.
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len, dtype=tf.float32)
        return positions + tf.cast(start_index, tf.float32)

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        rank = len(inputs.shape)
        # Here we assume the feature axis is the last axis.
        feature_axis = rank - 1
        sequence_axis = 1

        rotary_dim = tf.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
        else:
            positions = tf.cast(positions, tf.float32)

        positions = positions / tf.cast(self.scaling_factor, tf.float32)
        freq = tf.einsum("i,j->ij", positions, inverse_freq)
        # Duplicate frequency for sine and cosine.
        embedding = tf.stack([freq, freq], axis=-2)
        # Reshape so that the last dimension becomes twice its original size.
        freq_shape = tf.shape(freq)
        new_shape = tf.concat([freq_shape[:-1], [freq_shape[-1] * 2]], axis=0)
        embedding = tf.reshape(embedding, new_shape)

        if feature_axis < sequence_axis:
            embedding = tf.transpose(embedding)
        # Expand dimensions to broadcast over inputs if necessary.
        for axis in range(rank):
            if axis != sequence_axis and axis != feature_axis:
                embedding = tf.expand_dims(embedding, axis)
        cos_emb = tf.cast(tf.cos(embedding), self.compute_dtype)
        sin_emb = tf.cast(tf.sin(embedding), self.compute_dtype)
        return cos_emb, sin_emb

    def _get_inverse_freq(self, rotary_dim):
        # Ensure rotary_dim is treated as an integer for tf.range.
        rotary_dim_int = tf.cast(rotary_dim, tf.int32)
        freq_range = tf.range(0, rotary_dim_int, delta=2, dtype=tf.float32)
        freq_range = tf.divide(freq_range, tf.cast(rotary_dim, tf.float32))
        inverse_freq = 1.0 / (self.max_wavelength ** freq_range)
        return inverse_freq

    def get_config(self):
        config = super(RotaryEmbedding, self).get_config()
        config.update({
            "max_wavelength": self.max_wavelength,
            "scaling_factor": self.scaling_factor,
            "sequence_axis": self.sequence_axis,
            "feature_axis": self.feature_axis,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class RotaryEmbeddingMask(RotaryEmbedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, training=None, mask=None, **kwargs):
        ret = super().call(inputs, **kwargs)
        # Apply the mask to the output
        if mask is not None:
            ret._keras_mask = mask
        return ret


class BaseAttention(tf.keras.layers.Layer):
    """
    A basic multi-head attention layer
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        self.supports_masking = True

        def calculate_attention(x, y, mask):
            # if tf.get_static_value(tf.rank(mask)) == 0:
            #     mask = None

            attn_output = self.mha(
                query=x,
                value=y,
                key=y,
                attention_mask=mask
            )
            result = self.add([x, attn_output])
            return result

        self.calculate_attention = tf.recompute_grad(calculate_attention)

    def get_config(self):
        config = super().get_config()
        config.update({"mha": self.mha.get_config(),
                       "layer_norm": self.layer_norm.get_config(),
                       "add": self.add.get_config(),
                       "supports_masking": self.supports_masking})
        return config

    @staticmethod
    def generate_attention_mask(mask_x, mask_y):
        """Generate the cross-attention mask considering both mask_x and mask_y, with correct broadcasting for non-None masks.

        Args:
            mask_x: Mask for the query, shape [B, T], or None.
            mask_y: Mask for the key, shape [B, S], or None.

        Returns:
            A boolean Tensor representing the cross-attention mask with shape [B, T, S].
        """
        # Handle mask_x
        if mask_x is None:
            mask_x = tf.ones([1, 1, 1], dtype=tf.bool)  # Full attention if mask_x is missing
        else:
            mask_x = tf.cast(mask_x[:, :, tf.newaxis], dtype=tf.bool)  # Shape [B, T, 1]

        # Handle mask_y
        if mask_y is None:
            mask_y = tf.ones([1, 1, 1], dtype=tf.bool)  # Full attention if mask_y is missing
        else:
            mask_y = tf.cast(mask_y[:, tf.newaxis, :], dtype=tf.bool)  # Shape [B, 1, S]

        # Generate the combined mask with correct broadcasting, resulting in shape [B, T, S]
        combined_mask = tf.cast(tf.logical_and(mask_x, mask_y), dtype=tf.float32)

        return combined_mask


class SelfAttention(BaseAttention):
    """
    A self-attention layer, that using gradient check point to save memory
    """
    def compute_mask(self, inputs, mask=None):
        return mask

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs
        # calculate self-attention mask, mask is a function parameter generated by keras Mask has shape (batch_size,
        # seq_len) with boolean values, True means the token is masked, False means the token is not masked

        attention_mask = self.generate_attention_mask(mask, mask)
        x = self.calculate_attention(x, x, attention_mask)
        x = self.layer_norm(x)
        return x


class CrossAttention(BaseAttention):

    def compute_mask(self, inputs, mask=None):
        mask_x, mask_y = mask if mask is not None else (None, None)
        return mask_x

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        # Calculate cross-attention mask, mask is a function parameter generated by keras Mask has shape (batch_size,
        # seq_len) with boolean values, True means the token is masked, False means the token is not masked
        # Y is the main value, X is a query
        mask_x, mask_y = mask if mask is not None else (None, None)
        attention_mask = self.generate_attention_mask(mask_x, mask_y)

        y = self.calculate_attention(x, y, attention_mask)
        y = self.layer_norm(y)
        return y


class PositionEncoding1D(tf.keras.layers.Layer):
    def __init__(self, default_position=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.built = None
        self.pos_encoding = None
        self.d_model = None
        self.def_position = default_position
        with tf.init_scope():
            self.position = self.def_position
        self.mask = tf.keras.layers.Masking(mask_value=-1.0)

    def compute_mask(self, inputs, mask=None):
        return self.mask.compute_mask(inputs, mask)

    def get_config(self):
        config = super().get_config()
        config.update({"def_position": self.def_position})
        return config

    def build(self, input_shape):
        assert input_shape[-1] is not None, "Unit cannot undefined"
        self.d_model = input_shape[-1]
        if input_shape[-2] is None:
            self.position = self.def_position
        else:
            self.position = input_shape[-2]
        self.pos_encoding = self.positional_encoding()
        self.built = True

    def positional_encoding(self):
        angle_rates = 1 / tf.pow(10000.,
                                 tf.cast(tf.range(0, self.d_model, 2), self.dtype_policy.variable_dtype) / tf.cast(
                                     self.d_model, self.dtype_policy.variable_dtype))
        angle_rads = tf.einsum('i,j->ij', tf.cast(tf.range(self.position), self.dtype_policy.variable_dtype),
                               angle_rates)
        sin_cos = tf.math.sin(angle_rads)[..., tf.newaxis], tf.math.cos(angle_rads)[..., tf.newaxis]
        pos_encoding = tf.reshape(tf.concat(sin_cos, axis=-1), [angle_rads.shape[0], -1])[:, :self.d_model]
        return pos_encoding

    def get_encode(self, length, x, tig=5):
        encoder_msg = tf.cast(self.pos_encoding[0: length][tf.newaxis, ...], self.dtype_policy.compute_dtype)
        try:
            encode = encoder_msg + x
        except tf.errors.InvalidArgumentError:
            self.position = length if length > self.position * 2 else self.position * 2
            self.pos_encoding = self.positional_encoding()
            assert tig > 0, "Too much iteration, might caused by feature map size change"
            encode = self.get_encode(length, x, tig=tig - 1)
        return encode

    def call(self, x, training=False, mask=None):
        x_length = tf.shape(x)[-2]
        return self.get_encode(x_length, x)


# Define a Concatenate layer that will cut the input to the same length
class CutConcatenate(tf.keras.layers.Concatenate):

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        masks = []
        for mask_i in mask:
            if mask_i is not None:
                # If input mask is not None, cut the mask to match the first input's dimensions
                sliced_mask = tf.slice(mask_i, [0, 0], [-1, tf.shape(mask[0])[1]])
                masks.append(sliced_mask)
            else:
                masks.append(None)

        # Call super compute_mask which handles concatenation of masks and
        # returns a mask that corresponds to the minimum presence of 1s across all masks
        return super().compute_mask(masks)

    def call(self, inputs):
        inter = [tf.slice(x, [0, 0, 0], [-1, tf.shape(inputs[0])[1], -1]) for x in inputs]
        # Set shape of interring to the shape of inputs if input is Tensor Placeholder
        shapes = [x.shape for x in inputs]
        for y, x in zip(inter, inputs):
            d = [sx if sx is not None and sy is None else sy for sx, sy in zip(x.shape, y.shape)]
            y.set_shape(d)

        return super().call(inter)


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
        SwiGLU Activation Layer
        """
        super(SwiGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)
        self.supports_masking = True

        def swiGLU(x):
            out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
            gate = tf.keras.activations.swish(gate)
            x = tf.multiply(out, gate)
            return x

        self.swiGLU = tf.recompute_grad(swiGLU)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({"bias": self.bias, "dim": self.dim})
        return config

    def call(self, x, mask=None, **kwargs):
        x = self.swiGLU(x)
        return x
