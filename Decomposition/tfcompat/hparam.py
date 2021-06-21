"""Hyperparameter values."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numbers
import re

import six

PARAM_RE = re.compile(r"""
  (?P<name>[a-zA-Z][\w\.]*)      # variable name: "var" or "x"
  (\[\s*(?P<index>\d+)\s*\])?  # (optional) index: "1" or None
  \s*=\s*
  ((?P<val>[^,\[]*)            # single value: "a" or None
   |
   \[(?P<vals>[^\]]*)\])       # list of values: None or "1,2,3"
  ($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
  """Helper function for raising a value error for bad assignment."""
  raise ValueError(
      'Could not parse hparam \'%s\' of type \'%s\' with value \'%s\' in %s' %
      (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
  """Helper function for raising a value error for reuse of name."""
  raise ValueError('Multiple assignments to variable \'%s\' in %s' % (name,
                                                                      values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values,
                          results_dictionary):
  try:
    parsed_value = parse_fn(m_dict['val'])
  except ValueError:
    _parse_fail(name, var_type, m_dict['val'], values)

  # If no index is provided
  if not m_dict['index']:
    if name in results_dictionary:
      _reuse_fail(name, values)
    results_dictionary[name] = parsed_value
  else:
    if name in results_dictionary:
      if not isinstance(results_dictionary.get(name), dict):
        _reuse_fail(name, values)
    else:
      results_dictionary[name] = {}

    index = int(m_dict['index'])
    if index in results_dictionary[name]:
      _reuse_fail('{}[{}]'.format(name, index), values)
    results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values,
                        results_dictionary):
  if m_dict['index'] is not None:
    raise ValueError('Assignment of a list to a list index.')
  elements = filter(None, re.split('[ ,]', m_dict['vals']))
  if name in results_dictionary:
    raise _reuse_fail(name, values)
  try:
    results_dictionary[name] = [parse_fn(e) for e in elements]
  except ValueError:
    _parse_fail(name, var_type, m_dict['vals'], values)


def _cast_to_type_if_compatible(name, param_type, value):
  fail_msg = (
      "Could not cast hparam '%s' of type '%s' from value %r" %
      (name, param_type, value))

  if issubclass(param_type, type(None)):
    return value

  if (issubclass(param_type, (six.string_types, six.binary_type)) and
      not isinstance(value, (six.string_types, six.binary_type))):
    raise ValueError(fail_msg)

  if issubclass(param_type, bool) != isinstance(value, bool):
    raise ValueError(fail_msg)

  if (issubclass(param_type, numbers.Integral) and
      not isinstance(value, numbers.Integral)):
    raise ValueError(fail_msg)

  if (issubclass(param_type, numbers.Number) and
      not isinstance(value, numbers.Number)):
    raise ValueError(fail_msg)

  return param_type(value)


def parse_values(values, type_map):
  results_dictionary = {}
  pos = 0
  while pos < len(values):
    m = PARAM_RE.match(values, pos)
    if not m:
      raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
    
    pos = m.end()
    
    m_dict = m.groupdict()
    name = m_dict['name']
    if name not in type_map:
      raise ValueError('Unknown hyperparameter type for %s' % name)
    type_ = type_map[name]

    if type_ == bool:

      def parse_bool(value):
        if value in ['true', 'True']:
          return True
        elif value in ['false', 'False']:
          return False
        else:
          try:
            return bool(int(value))
          except ValueError:
            _parse_fail(name, type_, value, values)

      parse = parse_bool
    else:
      parse = type_

    if m_dict['val'] is not None:
      _process_scalar_value(name, parse, type_, m_dict, values,
                            results_dictionary)

    elif m_dict['vals'] is not None:
      _process_list_value(name, parse, type_, m_dict, values,
                          results_dictionary)

    else:
      _parse_fail(name, type_, '', values)

  return results_dictionary


class HParams(object):
    self._hparam_types = {}
    self._model_structure = model_structure
    if hparam_def:
      raise ValueError('hparam_def has been disabled in this version')
    else:
      for name, value in six.iteritems(kwargs):
        self.add_hparam(name, value)

  def add_hparam(self, name, value):
    
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError(
            'Multi-valued hyperparameters cannot be empty: %s' % name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)

  def set_hparam(self, name, value):
    param_type, is_list = self._hparam_types[name]
    if isinstance(value, list):
      if not is_list:
        raise ValueError(
            'Must not pass a list for single-valued parameter: %s' % name)
      setattr(self, name, [
          _cast_to_type_if_compatible(name, param_type, v) for v in value])
    else:
      if is_list:
        raise ValueError(
            'Must pass a list for multi-valued parameter: %s.' % name)
      setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

  def del_hparam(self, name):
    if hasattr(self, name):
      delattr(self, name)
      del self._hparam_types[name]

  def parse(self, values):
    type_map = dict()
    for name, t in self._hparam_types.items():
      param_type, _ = t
      type_map[name] = param_type

    values_map = parse_values(values, type_map)
    return self.override_from_dict(values_map)

  def override_from_dict(self, values_dict):
    for name, value in values_dict.items():
      self.set_hparam(name, value)
    return self

  def set_from_map(self, values_map):
    return self.override_from_dict(values_dict=values_map)

  def set_model_structure(self, model_structure):
    self._model_structure = model_structure

  def get_model_structure(self):
    return self._model_structure

  def to_json(self, indent=None, separators=None, sort_keys=False):
    
    return json.dumps(
        self.values(),
        indent=indent,
        separators=separators,
        sort_keys=sort_keys)

  def parse_json(self, values_json):
    
    values_map = json.loads(values_json)
    return self.override_from_dict(values_map)

  def values(self):
    return {n: getattr(self, n) for n in self._hparam_types.keys()}

  def get(self, key, default=None):
    if key in self._hparam_types:
      if default is not None:
        param_type, is_param_list = self._hparam_types[key]
        type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
        fail_msg = ("Hparam '%s' of type '%s' is incompatible with "
                    'default=%s' % (key, type_str, default))

        is_default_list = isinstance(default, list)
        if is_param_list != is_default_list:
          raise ValueError(fail_msg)

        try:
          if is_default_list:
            for value in default:
              _cast_to_type_if_compatible(key, param_type, value)
          else:
            _cast_to_type_if_compatible(key, param_type, default)
        except ValueError as e:
          raise ValueError('%s. %s' % (fail_msg, e))

      return getattr(self, key)

    return default

  def __contains__(self, key):
    return key in self._hparam_types

  def __str__(self):
    return str(sorted(self.values().items()))

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__, self.__str__())

  @staticmethod
  def _get_kind_name(param_type, is_list):
    if issubclass(param_type, bool):
      
      typename = 'bool'
    elif issubclass(param_type, six.integer_types):
      
      typename = 'int64'
    elif issubclass(param_type, (six.string_types, six.binary_type)):
      
      typename = 'bytes'
    elif issubclass(param_type, float):
      typename = 'float'
    else:
      raise ValueError('Unsupported parameter type: %s' % str(param_type))

    suffix = 'list' if is_list else 'value'
    return '_'.join([typename, suffix])
