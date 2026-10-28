"""Extract concrete registration helpers from the actual installed-candidate archive."""
import re
from zipfile import ZipFile


def inventory(archive):
    with ZipFile(archive) as z:
        source = z.read('tensorflow/lite/micro/micro_mutable_op_resolver.h').decode()
    methods = list(re.finditer(r'TfLiteStatus (Add\w+)\s*\(', source))
    rows = []
    for i, match in enumerate(methods):
        method = match[1]
        if method in ('AddBuiltin', 'AddCustom'):
            continue
        body = source[match.start():methods[i+1].start() if i+1 < len(methods) else len(source)]
        builtin = re.search(r'AddBuiltin\(\s*BuiltinOperator_(\w+)', body)
        custom = re.search(r'AddCustom\(\s*"([^"]+)"', body)
        if builtin:
            name, kind = builtin[1], 'builtin'
        elif custom:
            name, kind = custom[1], 'custom'
        elif method == 'AddEthosU':
            name, kind = 'ETHOS_U', 'custom'  # stable inventory key for GetString_ETHOSU()
        else:
            raise ValueError(f'Unrecognized registration helper: {method}')
        rows.append(dict(operator=name, kind=kind, method=method))
    if not rows or len({r['operator'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate operator inventory')
    return rows


def reconcile(rows, recipes, skips):
    names = {r['operator'] for r in rows}
    if set(recipes) & set(skips):
        raise ValueError('A recipe and skip overlap')
    if names != set(recipes) | set(skips):
        raise ValueError(f'Coverage gap/stale entry: missing={names-set(recipes)-set(skips)}, '
                         f'stale={(set(recipes)|set(skips))-names}')
    if any(not str(reason).strip() for reason in skips.values()):
        raise ValueError('Every skip needs a reason')
    return [dict(row, status='planned' if row['operator'] in recipes else 'skipped',
                 reason=skips.get(row['operator'], '')) for row in rows]
